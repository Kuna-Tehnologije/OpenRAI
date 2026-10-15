#include <openrai/primary_node.h>
#include <openrai/semantic.h>
#include <openrai/log.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace openrai {

// -----------------------------------------------------------------------
// small helpers
// -----------------------------------------------------------------------
const char* behaviour_action_name(BehaviourAction a) {
    switch (a) {
        case BehaviourAction::Start:   return "START";
        case BehaviourAction::Stop:    return "STOP";
        case BehaviourAction::Suspend: return "SUSPEND";
        case BehaviourAction::Resume:  return "RESUME";
        case BehaviourAction::Kill:    return "KILL";
        case BehaviourAction::Respawn: return "RESPAWN";
        case BehaviourAction::Promote: return "PROMOTE";
        case BehaviourAction::Demote:  return "DEMOTE";
    }
    return "?";
}

static std::string f2(float v) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << v;
    return ss.str();
}

size_t FiringPlan::fired_count() const {
    size_t n = 0;
    for (const auto& d : decisions) if (d.fire) ++n;
    return n;
}

std::string FiringPlan::summary() const {
    std::ostringstream ss;
    ss << "[PRIMARY] tick=" << tick
       << " focus=" << focus_goal << "(" << f2(focus_score) << ")"
       << " judge[rel=" << f2(judgement.relevance)
       << " imp=" << f2(judgement.importance)
       << " prio=" << f2(judgement.priority) << "]"
       << " concurrency=" << concurrency
       << " fire=" << fired_count() << "/" << decisions.size();

    ss << " FIRE[";
    bool first = true;
    for (const auto& d : decisions) {
        if (!d.fire) continue;
        if (!first) ss << "; ";
        first = false;
        ss << d.node_name << " " << f2(d.score) << " <- " << d.reason;
    }
    ss << "] HOLD[";
    first = true;
    for (const auto& d : decisions) {
        if (d.fire) continue;
        if (!first) ss << "; ";
        first = false;
        ss << d.node_name << " " << f2(d.score) << " <- " << d.reason;
    }
    ss << "]";
    return ss.str();
}

std::string Arbitration::summary() const {
    std::ostringstream ss;
    ss << "[ARBITER] tick=" << tick;
    if (has_winner) {
        ss << " WINNER=" << winner.node_name
           << " utility=" << f2(winner.utility)
           << " imp=" << f2(winner.best.importance)
           << " rel=" << f2(winner.best.relevance)
           << " latency_ms=" << std::fixed << std::setprecision(1) << (winner.latency.count() / 1000.0)
           << " because: " << reason;
    } else {
        ss << " NO WINNER: " << reason;
    }
    if (ranked.size() > 1) {
        ss << " | ranking=";
        for (size_t i = 0; i < ranked.size(); ++i) {
            if (i) ss << ",";
            ss << ranked[i].node_name << ":" << f2(ranked[i].utility);
        }
    }
    if (!late.empty()) {
        ss << " | LATE=";
        for (size_t i = 0; i < late.size(); ++i) {
            if (i) ss << ",";
            ss << late[i];
        }
    }
    return ss.str();
}

// -----------------------------------------------------------------------
// PrimaryNode
// -----------------------------------------------------------------------
PrimaryNode::PrimaryNode(const EngineConfig& cfg,
                         std::shared_ptr<GPUNeuralNet> judgement_net,
                         GoalSystem* goals)
    : cfg_(cfg),
      net_(std::move(judgement_net)),
      goals_(goals),
      adaptive_cap_(cfg.max_concurrent)
{
    // Highest priority stream of the whole engine: the executive never
    // waits behind an expert.
    ctx_ = std::make_unique<GPUStreamContext>("PRIMARY", 100, cfg_.node_max_batch, net_->arch());
    input_buffer_.assign(cfg_.node_max_batch * GPUNeuralNet::INPUT_DIM, 0.f);

    LOG("[PRIMARY] initialised"
        << " net=" << net_->arch().name << "[" << net_->arch().describe() << "]"
        << " params=" << net_->param_count()
        << " stream=" << (ctx_->is_cuda() ? "cuda" : "cpu")
        << " cuda_prio=" << ctx_->cuda_priority()
        << " max_concurrent=" << cfg_.max_concurrent
        << " fire_threshold=" << cfg_.fire_threshold);
}

PrimaryNode::~PrimaryNode() {
    if (ctx_) ctx_->synchronize();
}

float PrimaryNode::goal_weight(const std::string& goal) const {
    if (!goals_) return 0.33f;
    auto all = goals_->top_goals(16);
    float sum = 0.f, mine = 0.f;
    for (const auto& g : all) {
        sum += g.score;
        if (g.name == goal) mine = g.score;
    }
    if (sum <= 0.f) return 0.f;
    return mine / sum;
}

std::string PrimaryNode::focus_goal() const {
    if (!goals_) return "relevance";
    auto top = goals_->top_goals(1);
    return top.empty() ? "relevance" : top.front().name;
}

// ---- 1. judge --------------------------------------------------------------
NNResult PrimaryNode::judge(const std::string& input) {
    encode_input_v2(input, input_buffer_.data(), GPUNeuralNet::INPUT_DIM);
    if (!net_->run_batch_async(input_buffer_.data(), 1, *ctx_)) {
        LOG("[PRIMARY] judge: launch refused (context cancelled or mismatched)");
        return NNResult{};
    }
    ctx_->synchronize();
    std::vector<NNResult> r;
    net_->fetch_results(*ctx_, 1, r);
    return r.empty() ? NNResult{} : r.front();
}

// ---- 2. plan ---------------------------------------------------------------
FiringPlan PrimaryNode::plan(uint64_t tick,
                             const std::string& input,
                             const NNResult& judgement,
                             const NodeList& nodes,
                             std::chrono::steady_clock::time_point deadline,
                             bool external)
{
    FiringPlan p;
    p.tick      = tick;
    p.input     = input;
    p.judgement = judgement;
    p.deadline  = deadline;

    auto top = goals_ ? goals_->top_goals(1) : std::vector<GoalState>{};
    p.focus_goal  = top.empty() ? "relevance" : top.front().name;
    p.focus_score = top.empty() ? 0.f : top.front().score;

    // Breadth of this tick: the judgement's priority output widens the
    // search (more experts), bounded by the adaptive cap (shrinks after
    // overruns) and the configured maximum.
    const size_t lo = std::max<size_t>(1, cfg_.min_concurrent);
    const size_t hi = std::max(lo, std::min(cfg_.max_concurrent, adaptive_cap_));
    p.concurrency = lo + static_cast<size_t>(std::lround((hi - lo) * std::clamp(judgement.priority, 0.f, 1.f)));
    p.concurrency = std::clamp(p.concurrency, lo, hi);

    const double budget_ms = cfg_.compute_budget().count() / 1e6;

    struct Cand { size_t idx; float score; };
    std::vector<Cand> eligible;

    // ---- competence: ask every expert's model how familiar the input is ----
    // The scores are log-likelihoods per token; they are compared *across*
    // experts (relative to the most competent one), so the expert whose
    // model knows this kind of text best gets affinity 1.0 and the others
    // fall off exponentially.  Nodes without a trained model fall back to
    // their bootstrap keyword affinity.
    std::vector<ModelScore> comp(nodes.size());
    float best_lp = -1e30f;
    bool any_model = false;
    for (size_t i = 0; i < nodes.size(); ++i) {
        comp[i] = nodes[i]->competence(input);
        if (comp[i].valid) { any_model = true; best_lp = std::max(best_lp, comp[i].mean_logprob); }
    }
    auto model_affinity = [&](size_t i) -> float {
        if (!comp[i].valid) return nodes[i]->keyword_affinity(input);
        // one nat behind the best expert -> 0.37, three nats -> 0.05
        const float rel = std::exp(std::max(-20.f, comp[i].mean_logprob - best_lp));
        // unknown words weaken the claim ("I have seen words like these")
        return std::clamp(rel * (0.5f + 0.5f * comp[i].coverage), 0.f, 1.f);
    };
    (void)any_model;

    p.decisions.reserve(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        const ExpertNode& n = *nodes[i];
        const NodeStats st  = n.stats();
        FiringDecision d;
        d.node_id   = n.id();
        d.node_name = n.name();
        d.affinity  = model_affinity(i);

        // goal share of this node's goal; 1/3 when goals are balanced,
        // rescaled so balanced -> 0.5 and a dominant goal -> ~1.0
        const float gw       = std::min(1.f, goal_weight(n.domain().goal) * 1.5f);
        const bool  on_focus = n.domain().goal == p.focus_goal;
        const float starve   = st.last_fired_tick == 0
                             ? 1.f
                             : std::min(1.f, static_cast<float>(tick - st.last_fired_tick) /
                                             static_cast<float>(std::max(1, cfg_.starvation_ticks)));
        const bool  slow     = st.ema_latency_ms > budget_ms;

        float score = 0.35f * d.affinity
                    + 0.25f * gw
                    + 0.25f * st.trust
                    + 0.15f * starve
                    + (on_focus ? 0.10f : 0.f);
        if (slow) score -= 0.30f;
        // A user question: experts that can actually solve or recall
        // something (skills) matter more than relevance scorers.
        const bool skilled = n.skill() != nullptr && !n.is_slow();
        if (external && skilled) score += 0.20f;
        score = std::clamp(score, 0.f, 1.f);
        d.score = score;

        // Reason string - the strongest contributors first
        std::ostringstream why;
        if (on_focus)         why << "focus-goal=" << p.focus_goal << " ";
        if (comp[i].valid) {
            if (d.affinity > 0.5f)
                why << "model-competence=" << f2(d.affinity) << "(logp " << std::fixed << std::setprecision(2)
                    << comp[i].mean_logprob << ") ";
        } else if (d.affinity > 0.5f) {
            why << "keyword-affinity=" << f2(d.affinity) << "(no model) ";
        }
        if (st.trust > 0.6f)  why << "trusted=" << f2(st.trust) << " ";
        if (starve >= 1.f && st.fired > 0) why << "starved=" << (tick - st.last_fired_tick) << "t ";
        if (st.fired == 0)    why << "never-fired ";
        if (external && skilled) why << "skill=" << n.skill()->name << " for user question ";
        if (slow)             why << "SLOW(ema=" << std::fixed << std::setprecision(0) << st.ema_latency_ms << "ms>" << budget_ms << ") ";

        const NodeState s = n.state();
        if (n.is_slow()) {
            d.fire   = false;
            d.reason = s == NodeState::Running ? "stage 2 expert, busy phrasing an earlier answer"
                                               : "stage 2 expert: fired after arbitration for user questions";
            p.decisions.push_back(std::move(d));
            continue;
        }
        if (s != NodeState::Idle) {
            d.fire = false;
            d.reason = std::string("not eligible: ") + node_state_name(s);
            if (s == NodeState::Running)   d.reason += " (still busy from tick " + std::to_string(st.last_fired_tick) + ")";
            if (s == NodeState::Suspended) d.reason += " (low utility, trust=" + f2(st.trust) + ")";
            if (s == NodeState::Killed)    d.reason += " (" + std::to_string(st.consecutive_misses) + " deadline misses)";
            p.decisions.push_back(std::move(d));
            continue;
        }

        d.reason = why.str();
        while (!d.reason.empty() && d.reason.back() == ' ') d.reason.pop_back();
        if (d.reason.empty()) d.reason = "baseline";
        eligible.push_back({ p.decisions.size(), score });
        p.decisions.push_back(std::move(d));
    }

    std::sort(eligible.begin(), eligible.end(),
              [](const Cand& a, const Cand& b) { return a.score > b.score; });

    size_t fired = 0;
    // Experts that must be consulted for every user question (the memory
    // expert: "have we answered this before?") fire outside the budget.
    if (external) {
        for (auto& c : eligible) {
            FiringDecision& d = p.decisions[c.idx];
            const ExpertNode& n = *nodes[static_cast<size_t>(d.node_id)];
            if (n.domain().consult_for_questions && !d.fire) {
                d.fire = true;
                d.reason = "consulted for every user question (" + d.reason + ")";
            }
        }
    }
    for (size_t k = 0; k < eligible.size(); ++k) {
        FiringDecision& d = p.decisions[eligible[k].idx];
        if (d.fire) continue;
        if (fired >= p.concurrency) {
            d.reason = "budget: concurrency " + std::to_string(p.concurrency) + " reached (score " + f2(d.score) + ")";
            continue;
        }
        if (d.score < cfg_.fire_threshold && fired > 0) {
            d.reason = "score " + f2(d.score) + " < threshold " + f2(cfg_.fire_threshold);
            continue;
        }
        if (d.score < cfg_.fire_threshold && fired == 0) {
            d.reason = "fallback: best available (" + d.reason + ")";
        }
        d.fire = true;
        ++fired;
    }

    return p;
}

// ---- 3. arbitrate ------------------------------------------------------------
Arbitration PrimaryNode::arbitrate(uint64_t tick,
                                   std::vector<NodeOutcome>& outcomes,
                                   const FiringPlan& plan,
                                   NodeList& nodes)
{
    (void)plan;
    Arbitration a;
    a.tick = tick;

    auto node_by_id = [&](int id) -> ExpertNode* {
        for (auto& n : nodes) if (n->id() == id) return n.get();
        return nullptr;
    };

    for (auto& o : outcomes) {
        ExpertNode* n = node_by_id(o.node_id);
        if (!n) continue;

        if (o.cancelled || !o.launched || o.results.empty()) {
            a.late.push_back(o.node_name + "(cancelled)");
            continue;
        }
        if (!o.on_time) {
            a.late.push_back(o.node_name + "(" + std::to_string(o.latency.count() / 1000) + "ms)");
            n->update_stats([&](NodeStats& st) {
                ++st.deadline_misses;
                ++st.consecutive_misses;
                st.trust = std::max(0.f, st.trust - 0.10f);
            });
            continue;
        }

        const NodeStats st = n->stats();
        const float gw = goal_weight(o.goal);
        float u = 0.40f * o.best.importance
                + 0.20f * o.best.relevance
                + 0.20f * o.affinity
                + 0.20f * std::min(1.f, gw * 1.5f);
        u *= (0.6f + 0.4f * st.trust);
        if (o.best.should_out) u += 0.05f;
        if (o.skill.handled) {
            // A deterministic expert that actually solved the input beats
            // any mere relevance score.
            u = std::max(u, 0.55f + 0.45f * std::clamp(o.skill.confidence, 0.f, 1.f));
        }
        o.utility = std::clamp(u, 0.f, 1.f);
        a.ranked.push_back(o);

        n->update_stats([&](NodeStats& s) {
            s.consecutive_misses = 0;
            s.last_utility = o.utility;
            s.trust = std::clamp(s.trust + cfg_.trust_alpha * (o.utility - s.trust), 0.f, 1.f);
        });
    }

    std::sort(a.ranked.begin(), a.ranked.end(),
              [](const NodeOutcome& x, const NodeOutcome& y) { return x.utility > y.utility; });

    if (a.ranked.empty()) {
        a.has_winner = false;
        a.reason = outcomes.empty() ? "no expert fired" : "every fired expert missed the deadline";
        return a;
    }

    a.has_winner = true;
    a.winner = a.ranked.front();
    if (ExpertNode* w = node_by_id(a.winner.node_id)) {
        w->update_stats([&](NodeStats& s) {
            ++s.wins;
            s.last_win_tick = tick;
            s.trust = std::min(1.f, s.trust + 0.05f);
        });
    }

    std::ostringstream why;
    if (a.winner.skill.handled)
        why << "SOLVED by skill (confidence " << f2(a.winner.skill.confidence) << ", answer \""
            << a.winner.skill.answer << "\"); ";
    why << "importance " << f2(a.winner.best.importance)
        << " x goal(" << a.winner.goal << ")=" << f2(goal_weight(a.winner.goal))
        << " affinity " << f2(a.winner.affinity)
        << (a.winner.best.should_out ? " should_out" : "")
        << " fired-for \"" << a.winner.fire_reason << "\"";
    if (a.ranked.size() > 1)
        why << " beat " << a.ranked[1].node_name << " by " << f2(a.winner.utility - a.ranked[1].utility);
    a.reason = why.str();
    return a;
}

void PrimaryNode::note_deadline_miss(ExpertNode& node, uint64_t tick) {
    node.update_stats([&](NodeStats& st) {
        ++st.deadline_misses;
        ++st.consecutive_misses;
        st.trust = std::max(0.f, st.trust - 0.10f);
    });
    (void)tick;
}

// ---- 4. review (behaviour policy) ----------------------------------------------
std::vector<BehaviourDecision> PrimaryNode::review(uint64_t tick,
                                                   NodeList& nodes,
                                                   const Arbitration& arb,
                                                   bool tick_overran)
{
    std::vector<BehaviourDecision> out;
    const std::string focus = focus_goal();

    // Adaptive breadth: shrink after an overrun, grow back slowly.
    if (tick_overran) {
        ++overruns_;
        on_time_streak_ = 0;
        if (adaptive_cap_ > std::max<size_t>(1, cfg_.min_concurrent)) {
            --adaptive_cap_;
            LOG("[PRIMARY] tick=" << tick << " overrun -> concurrency cap lowered to " << adaptive_cap_);
        }
    } else if (++on_time_streak_ >= 8 && adaptive_cap_ < cfg_.max_concurrent) {
        ++adaptive_cap_;
        on_time_streak_ = 0;
        LOG("[PRIMARY] tick=" << tick << " stable -> concurrency cap raised to " << adaptive_cap_);
    }

    size_t idle_or_running = 0;
    for (auto& n : nodes) {
        const NodeState s = n->state();
        if (s == NodeState::Idle || s == NodeState::Running) ++idle_or_running;
    }

    for (auto& np : nodes) {
        ExpertNode& n = *np;
        const NodeStats st = n.stats();
        const NodeState s  = n.state();
        const bool on_focus = n.domain().goal == focus;
        const uint64_t since = tick >= st.state_since_tick ? tick - st.state_since_tick : 0;

        BehaviourDecision d;
        d.node_id   = n.id();
        d.node_name = n.name();

        switch (s) {
        case NodeState::Running:
            // Still running past the deadline.  Runaway?
            if (st.consecutive_misses >= static_cast<uint64_t>(cfg_.kill_after_misses)) {
                d.action = BehaviourAction::Kill;
                d.reason = "runaway: " + std::to_string(st.consecutive_misses) + " consecutive deadline misses, ema_latency="
                         + f2(static_cast<float>(st.ema_latency_ms)) + "ms";
                out.push_back(d);
            }
            break;

        case NodeState::Idle:
            if (st.consecutive_misses >= static_cast<uint64_t>(cfg_.kill_after_misses)) {
                // Returned eventually but chronically late: restart cold
                d.action = BehaviourAction::Kill;
                d.reason = "chronically late: " + std::to_string(st.consecutive_misses) + " consecutive misses";
                out.push_back(d);
                break;
            }
            if (st.trust < cfg_.suspend_trust && !on_focus && idle_or_running > 1) {
                const int streak = st.low_trust_streak + 1;
                n.update_stats([&](NodeStats& x) { x.low_trust_streak = streak; });
                if (streak >= cfg_.suspend_patience) {
                    d.action = BehaviourAction::Suspend;
                    d.reason = "low utility: trust " + f2(st.trust) + " < " + f2(cfg_.suspend_trust)
                             + " for " + std::to_string(streak) + " ticks";
                    out.push_back(d);
                    --idle_or_running;
                    break;
                }
            } else if (st.low_trust_streak != 0) {
                n.update_stats([&](NodeStats& x) { x.low_trust_streak = 0; });
            }
            break;

        case NodeState::Suspended:
            if (on_focus) {
                d.action = BehaviourAction::Resume;
                d.reason = "goal demands it: focus=" + focus;
                out.push_back(d);
            } else if (since >= static_cast<uint64_t>(cfg_.probation_ticks)) {
                d.action = BehaviourAction::Resume;
                d.reason = "probation over (" + std::to_string(since) + " ticks), re-evaluating";
                n.update_stats([&](NodeStats& x) { x.trust = std::max(x.trust, cfg_.suspend_trust + 0.10f); });
                out.push_back(d);
            }
            break;

        case NodeState::Killed:
            if (since >= static_cast<uint64_t>(cfg_.respawn_after_ticks) || (on_focus && since >= 2)) {
                d.action = BehaviourAction::Respawn;
                d.reason = on_focus ? "focus goal " + focus + " needs this expert"
                                    : "cooldown over (" + std::to_string(since) + " ticks)";
                out.push_back(d);
            }
            break;

        case NodeState::Stopped:
            if (on_focus || since >= static_cast<uint64_t>(cfg_.respawn_after_ticks)) {
                d.action = BehaviourAction::Start;
                d.reason = on_focus ? "focus goal " + focus + " needs this expert" : "cold start";
                out.push_back(d);
            }
            break;
        }

        // Priority follows trust and goal weight; re-prioritise only on
        // significant change (recreating a stream is not free).
        if (s == NodeState::Idle || s == NodeState::Suspended) {
            const float gw = std::min(1.f, goal_weight(n.domain().goal) * 1.5f);
            const int desired = std::clamp(static_cast<int>(std::lround(30.0 + 70.0 * (0.5 * st.trust + 0.5 * gw))), 0, 100);
            const int cur = n.priority();
            if (std::abs(desired - cur) >= 15) {
                BehaviourDecision pd;
                pd.node_id      = n.id();
                pd.node_name    = n.name();
                pd.action       = desired > cur ? BehaviourAction::Promote : BehaviourAction::Demote;
                pd.new_priority = desired;
                pd.reason       = "trust " + f2(st.trust) + " goal_w " + f2(gw) + " -> priority " + std::to_string(desired);
                out.push_back(pd);
            }
        }
    }

    // Winner gets a nudge upwards so its stream wins ties on the GPU
    if (arb.has_winner) {
        for (auto& np : nodes) {
            if (np->id() != arb.winner.node_id) continue;
            const int cur = np->priority();
            if (cur < 85 && np->stats().wins % 4 == 0) {
                BehaviourDecision pd;
                pd.node_id      = np->id();
                pd.node_name    = np->name();
                pd.action       = BehaviourAction::Promote;
                pd.new_priority = std::min(100, cur + 10);
                pd.reason       = "won arbitration (" + std::to_string(np->stats().wins) + " wins)";
                out.push_back(pd);
            }
        }
    }

    return out;
}

void PrimaryNode::apply(const std::vector<BehaviourDecision>& decisions,
                        NodeList& nodes, uint64_t tick)
{
    for (const auto& d : decisions) {
        ExpertNode* n = nullptr;
        for (auto& np : nodes) if (np->id() == d.node_id) { n = np.get(); break; }
        if (!n) continue;

        bool ok = true;
        switch (d.action) {
        case BehaviourAction::Start:
        case BehaviourAction::Respawn: ok = n->start(tick);   break;
        case BehaviourAction::Stop:    n->stop(tick);         break;
        case BehaviourAction::Suspend: ok = n->suspend(tick); break;
        case BehaviourAction::Resume:  ok = n->resume(tick);  break;
        case BehaviourAction::Kill:    n->kill(tick);         break;
        case BehaviourAction::Promote:
        case BehaviourAction::Demote:  ok = n->set_priority(d.new_priority, tick); break;
        }

        LOG("[BEHAVIOUR] tick=" << tick << " " << behaviour_action_name(d.action)
            << " " << d.node_name
            << (d.new_priority >= 0 ? " priority=" + std::to_string(d.new_priority) : "")
            << " because: " << d.reason
            << (ok ? "" : " [deferred]"));
    }
}

} // namespace openrai
