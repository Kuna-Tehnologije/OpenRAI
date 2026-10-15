// -----------------------------------------------------------------------
// node_tests.cpp
//   Self-contained tests for the node scheduler:
//     * TickScheduler keeps 7.82 Hz and reports overruns
//     * GPUStreamContext / run_batch_async == run_batch
//     * ExpertNode lifecycle: start / fire / collect / suspend / kill / restart
//     * PrimaryNode plan / arbitrate / review behaviour policy
//     * Full engine smoke run at 7.82 Hz
//   Build: cmake -DOPENRAI_BUILD_TESTS=ON .. && ./openrai_tests
// -----------------------------------------------------------------------

#include <openrai/core.h>
#include <openrai/config.h>
#include <openrai/expert_node.h>
#include <openrai/primary_node.h>
#include <openrai/scheduler.h>
#include <openrai/semantic.h>
#include <openrai/skills.h>
#include <openrai/experience_store.h>
#include <openrai/lm_adapter.h>
#include <openrai/training_demo.h>
#include <openrai/corpus.h>
#include <openrai/expert_model.h>
#include <openrai/gpu_runtime.h>
#include <openrai/log.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace openrai;

static int g_failures = 0;
static int g_checks   = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        ++g_checks;                                                          \
        if (!(cond)) {                                                       \
            ++g_failures;                                                    \
            std::cerr << "  FAIL " << __FILE__ << ":" << __LINE__            \
                      << "  " #cond "\n";                                    \
        }                                                                    \
    } while (0)

#define SECTION(name) std::cout << "\n[TEST] " << name << std::endl

static EngineConfig test_config() {
    EngineConfig c;
    c.profile        = "lite";
    c.tick_hz        = 7.82;
    c.expert_count   = 4;
    c.max_concurrent = 2;
    c.node_max_batch = 4;
    c.primary_arch   = NetArch::parse("32,256,128,4", "test_primary");
    c.expert_arch    = NetArch::parse("32,256,128,4", "test_expert");
    c.run_seconds    = 2;
    c.kill_after_misses  = 2;
    c.suspend_patience   = 2;
    c.probation_ticks    = 4;
    c.respawn_after_ticks = 3;
    c.llm_backend    = "off";
    c.store_path     = "data/test_engine.db";
    c.memory_path    = "data/test_memory.db";
    c.corpus_dir     = "data/test_corpus";
    c.model_dir      = "data/test_models";     // trained once, then loaded
    c.train_iterations = 80;
    std::error_code ec;
    std::filesystem::remove(c.store_path, ec);
    std::filesystem::remove(c.store_path + "-wal", ec);
    std::filesystem::remove(c.store_path + "-shm", ec);
    std::filesystem::remove(c.memory_path, ec);
    return c;
}

// ---------------------------------------------------------------------------
static void test_scheduler() {
    SECTION("TickScheduler 7.82 Hz");
    EngineConfig c = test_config();
    TickScheduler s(c.tick_period(), c.compute_budget());
    CHECK(std::fabs(s.target_hz() - 7.82) < 0.01);

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 12; ++i) {
        auto t = s.begin_tick();
        CHECK(t.index == static_cast<uint64_t>(i + 1));
        CHECK(t.compute_deadline > t.start);
        CHECK(t.compute_deadline <= t.end);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        auto over = s.end_tick();
        CHECK(over.count() == 0);
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    // 12 ticks at 127.877 ms = 1534 ms; allow generous scheduling jitter
    CHECK(elapsed >= 1500 && elapsed <= 1700);
    CHECK(std::fabs(s.measured_hz() - 7.82) < 0.4);
    CHECK(s.overruns() == 0);

    // Overrun: a tick that takes longer than one period
    s.begin_tick();
    std::this_thread::sleep_for(std::chrono::milliseconds(140));
    auto over = s.end_tick();
    CHECK(over.count() > 0);
    CHECK(s.overruns() == 1);
    std::cout << "  measured_hz=" << s.measured_hz() << " overruns=" << s.overruns() << std::endl;
}

// ---------------------------------------------------------------------------
static void test_stream_context() {
    SECTION("GPUStreamContext async == sync");
    NetArch a = NetArch::parse("32,128,64,4", "tiny");
    CHECK(a.valid());
    CHECK(a.param_count() == 32u * 128 + 128 + 128u * 64 + 64 + 64u * 4 + 4);
    CHECK(NetArch::full_4b().param_count() > 3000000000ull);   // 3.06B

    GPUNeuralNet net(a, 7);
    GPUStreamContext ctx("test", 80, 4, a);
    CHECK(ctx.max_batch() == 4);
    CHECK(ctx.priority() == 80);
    CHECK(ctx.poll());

    std::vector<float> in(4 * 32);
    for (size_t i = 0; i < 4; ++i)
        encode_input_v2("sample input number " + std::to_string(i), in.data() + i * 32, 32);

    std::vector<NNResult> sync_r, async_r;
    net.run_batch(in.data(), 4, sync_r);
    CHECK(net.run_batch_async(in.data(), 4, ctx));
    ctx.synchronize();
    CHECK(ctx.poll());
    net.fetch_results(ctx, 4, async_r);
    CHECK(sync_r.size() == 4 && async_r.size() == 4);
    for (size_t i = 0; i < 4; ++i) {
        CHECK(std::fabs(sync_r[i].relevance  - async_r[i].relevance)  < 1e-5f);
        CHECK(std::fabs(sync_r[i].importance - async_r[i].importance) < 1e-5f);
        CHECK(sync_r[i].should_out == async_r[i].should_out);
    }
    CHECK(ctx.launches() == 1);

    // Too large batch / cancelled context are refused
    CHECK(!net.run_batch_async(in.data(), 5, ctx));
    ctx.cancel();
    CHECK(ctx.cancelled());
    CHECK(!net.run_batch_async(in.data(), 1, ctx));
    ctx.reset_cancel();
    CHECK(net.run_batch_async(in.data(), 1, ctx));
    ctx.synchronize();

    // Two contexts on one network (shared weights) give identical answers
    GPUStreamContext ctx2("test2", 20, 4, a);
    CHECK(net.run_batch_async(in.data(), 4, ctx2));
    ctx2.synchronize();
    std::vector<NNResult> r2;
    net.fetch_results(ctx2, 4, r2);
    for (size_t i = 0; i < 4; ++i)
        CHECK(std::fabs(r2[i].importance - sync_r[i].importance) < 1e-5f);
}

// ---------------------------------------------------------------------------
static void test_expert_node_lifecycle() {
    SECTION("ExpertNode lifecycle");
    NetArch a = NetArch::parse("32,128,64,4", "tiny");
    auto net = std::make_shared<GPUNeuralNet>(a, 3);
    NodeSignal signal;
    ExpertNode node(0, make_expert_domain(0), net, 4, 50, &signal);

    CHECK(node.state() == NodeState::Stopped);
    CHECK(node.name() == "E0:stability");
    CHECK(node.keyword_affinity("alert: error and crash on node 3") > 0.8f);
    CHECK(node.keyword_affinity("hello world") == 0.f);
    CHECK(!node.competence("anything").valid);        // no model yet

    // fire() is refused while stopped
    FireRequest r;
    r.tick = 1; r.inputs = { "alert: error" };
    r.deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    CHECK(!node.fire(r));

    CHECK(node.start(1));
    CHECK(node.state() == NodeState::Idle);
    CHECK(node.context() != nullptr);

    CHECK(node.fire(r));
    NodeOutcome o;
    for (int i = 0; i < 200 && !node.try_collect(o); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    CHECK(o.tick == 1);
    CHECK(o.results.size() == 1);
    CHECK(o.on_time);
    CHECK(!o.cancelled);
    CHECK(node.state() == NodeState::Idle);
    CHECK(node.stats().fired == 1);
    CHECK(node.stats().completed == 1);

    // batch of 3
    r.tick = 2; r.inputs = { "a", "b", "c" };
    CHECK(node.fire(r));
    for (int i = 0; i < 200 && !node.try_collect(o); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    CHECK(o.results.size() == 3);
    CHECK(o.best_index < 3);

    // suspend / resume
    CHECK(node.suspend(3));
    CHECK(node.state() == NodeState::Suspended);
    CHECK(!node.fire(r));
    CHECK(node.resume(4));
    CHECK(node.state() == NodeState::Idle);

    // priority change re-creates the stream while idle
    CHECK(node.set_priority(90, 4));
    CHECK(node.priority() == 90);
    CHECK(node.context()->priority() == 90);

    // stop -> cold; start again
    node.stop(5);
    CHECK(node.state() == NodeState::Stopped);
    CHECK(node.context() == nullptr);
    CHECK(node.start(6));
    CHECK(node.state() == NodeState::Idle);

    // kill while running a deliberately heavy network
    NetArch heavy = NetArch::parse("32,4096,4096,4096,4096,4", "heavy");
    auto heavy_net = std::make_shared<GPUNeuralNet>(heavy, 5);
    ExpertNode slow(1, make_expert_domain(1), heavy_net, 4, 50, &signal);
    CHECK(slow.start(1));
    r.tick = 7; r.inputs = { "x", "y", "z", "w" };
    r.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1);
    CHECK(slow.fire(r));
    CHECK(slow.state() == NodeState::Running);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    const auto k0 = std::chrono::steady_clock::now();
    slow.kill(8);
    const auto kill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - k0).count();
    CHECK(slow.state() == NodeState::Killed);
    CHECK(slow.context() == nullptr);
    CHECK(slow.stats().kills == 1);
    CHECK(!slow.fire(r));
    std::cout << "  kill() returned in " << kill_ms << " ms (cancelled between layers)" << std::endl;

    // respawn after kill
    CHECK(slow.start(9));
    CHECK(slow.state() == NodeState::Idle);
    CHECK(slow.stats().restarts == 1);
    r.tick = 10; r.inputs = { "x" };
    r.deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1);
    CHECK(slow.fire(r));
    for (int i = 0; i < 5000 && !slow.try_collect(o); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    CHECK(o.tick == 10);
    CHECK(!o.on_time);          // 1 ms deadline is impossible for the heavy net
    CHECK(!o.cancelled);
    slow.stop(11);
}

// ---------------------------------------------------------------------------
static void test_primary_node() {
    SECTION("PrimaryNode plan / arbitrate / review");
    EngineConfig c = test_config();
    GoalSystem goals;
    auto pnet = std::make_shared<GPUNeuralNet>(c.primary_arch, 42);
    PrimaryNode primary(c, pnet, &goals);

    NodeSignal signal;
    NodeList nodes;
    for (size_t i = 0; i < c.expert_count; ++i) {
        auto net = std::make_shared<GPUNeuralNet>(c.expert_arch, 100 + static_cast<unsigned>(i));
        nodes.push_back(std::make_unique<ExpertNode>(static_cast<int>(i), make_expert_domain(i), net,
                                                     c.node_max_batch, 50, &signal));
        nodes.back()->start(0);
    }
    // nodes: 0 stability, 1 anomaly, 2 memory, 3 throughput

    // Make "memory" the dominant goal
    for (int i = 0; i < 10; ++i) goals.update_from_text("memory context history retrieve");
    CHECK(primary.focus_goal() == "memory");
    CHECK(primary.goal_weight("memory") > primary.goal_weight("stability"));

    const std::string input = "vector: similarity search over prior failures for node 3";
    NNResult j; j.priority = 1.0f;   // widest search -> concurrency = max
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    FiringPlan plan = primary.plan(1, input, j, nodes, deadline);
    CHECK(plan.concurrency == c.max_concurrent);
    CHECK(plan.fired_count() == c.max_concurrent);
    CHECK(plan.decisions.size() == nodes.size());
    // memory expert: focus goal + high affinity -> must fire
    CHECK(plan.decisions[2].fire);
    CHECK(plan.decisions[2].reason.find("focus-goal=memory") != std::string::npos);
    CHECK(plan.decisions[2].affinity > 0.8f);
    for (const auto& d : plan.decisions) CHECK(!d.reason.empty());
    std::cout << "  " << plan.summary() << std::endl;

    // Narrow search
    j.priority = 0.0f;
    FiringPlan narrow = primary.plan(2, input, j, nodes, deadline);
    CHECK(narrow.concurrency == 1);
    CHECK(narrow.fired_count() == 1);

    // Busy nodes are not eligible
    FireRequest r; r.tick = 3; r.inputs = { input }; r.deadline = deadline;
    nodes[2]->suspend(3);
    FiringPlan p3 = primary.plan(3, input, j, nodes, deadline);
    CHECK(!p3.decisions[2].fire);
    CHECK(p3.decisions[2].reason.find("SUSPENDED") != std::string::npos);
    nodes[2]->resume(3);

    // Arbitration: fire two experts for real and pick the best
    std::vector<NodeOutcome> outcomes;
    for (int id : { 2, 1 }) {
        FireRequest q; q.tick = 4; q.inputs = { input }; q.affinity = nodes[id]->keyword_affinity(input);
        q.deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1); q.reason = "test";
        CHECK(nodes[id]->fire(q));
        NodeOutcome o;
        for (int i = 0; i < 2000 && !nodes[id]->try_collect(o); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        outcomes.push_back(o);
    }
    FiringPlan p4 = primary.plan(4, input, j, nodes, deadline);
    Arbitration arb = primary.arbitrate(4, outcomes, p4, nodes);
    CHECK(arb.has_winner);
    CHECK(arb.ranked.size() == 2);
    CHECK(arb.ranked[0].utility >= arb.ranked[1].utility);
    CHECK(arb.winner.node_id == arb.ranked[0].node_id);
    CHECK(!arb.reason.empty());
    CHECK(nodes[arb.winner.node_id]->stats().wins == 1);
    std::cout << "  " << arb.summary() << std::endl;

    // A late outcome is not ranked and costs trust
    NodeOutcome late = outcomes[1];
    late.on_time = false;
    const float trust_before = nodes[late.node_id]->stats().trust;
    std::vector<NodeOutcome> v{ late };
    Arbitration arb2 = primary.arbitrate(5, v, p4, nodes);
    CHECK(!arb2.has_winner);
    CHECK(arb2.late.size() == 1);
    CHECK(nodes[late.node_id]->stats().trust < trust_before);
    CHECK(nodes[late.node_id]->stats().consecutive_misses == 1);

    // Behaviour: low trust on a non-focus node -> Suspend after patience
    nodes[3]->update_stats([](NodeStats& s) { s.trust = 0.05f; s.low_trust_streak = 0; });
    std::vector<BehaviourDecision> dec;
    for (int t = 6; t < 6 + c.suspend_patience; ++t) dec = primary.review(t, nodes, arb, false);
    bool suspended = false;
    for (const auto& d : dec) if (d.node_id == 3 && d.action == BehaviourAction::Suspend) suspended = true;
    CHECK(suspended);
    primary.apply(dec, nodes, 9);
    CHECK(nodes[3]->state() == NodeState::Suspended);

    // Behaviour: chronic deadline misses -> Kill, then Respawn after cooldown
    nodes[1]->update_stats([&](NodeStats& s) { s.consecutive_misses = c.kill_after_misses; });
    dec = primary.review(10, nodes, arb, false);
    bool killed = false;
    for (const auto& d : dec) if (d.node_id == 1 && d.action == BehaviourAction::Kill) killed = true;
    CHECK(killed);
    primary.apply(dec, nodes, 10);
    CHECK(nodes[1]->state() == NodeState::Killed);
    CHECK(nodes[1]->context() == nullptr);

    // Not yet: cooldown not over, probation not over
    dec = primary.review(11, nodes, arb, false);
    for (const auto& d : dec) {
        CHECK(!(d.node_id == 1 && d.action == BehaviourAction::Respawn));
        CHECK(!(d.node_id == 3 && d.action == BehaviourAction::Resume));
    }

    // tick 13: cooldown (3) over for the killed node and probation (4)
    // over for the suspended one
    dec = primary.review(10 + c.respawn_after_ticks, nodes, arb, false);
    bool respawned = false, resumed = false;
    for (const auto& d : dec) {
        if (d.node_id == 1 && d.action == BehaviourAction::Respawn) respawned = true;
        if (d.node_id == 3 && d.action == BehaviourAction::Resume)  resumed = true;
    }
    CHECK(respawned);
    CHECK(resumed);
    primary.apply(dec, nodes, 10 + c.respawn_after_ticks);
    CHECK(nodes[1]->state() == NodeState::Idle);
    CHECK(nodes[1]->context() != nullptr);
    CHECK(nodes[1]->stats().restarts == 1);
    CHECK(nodes[3]->state() == NodeState::Idle);

    // Adaptive concurrency shrinks on overrun
    const size_t cap0 = primary.adaptive_concurrency();
    primary.review(40, nodes, arb, true);
    CHECK(primary.adaptive_concurrency() == cap0 - 1);

    for (auto& n : nodes) n->stop(99);
}

// ---------------------------------------------------------------------------
static void test_engine_smoke() {
    SECTION("Engine smoke run @ 7.82 Hz");
    EngineConfig c = test_config();
    c.run_seconds = 2;
    {
        OpenRAI engine(c);
        engine.inject_input("alert: memory usage exceeded 90 percent on node 2");
        engine.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        engine.stop();
        auto st = engine.stats();
        std::cout << "  ticks=" << st.ticks << " hz=" << st.measured_hz
                  << " avg_busy_ms=" << st.avg_busy_ms
                  << " overruns=" << st.overruns << " wins=" << st.wins << std::endl;
        CHECK(st.ticks >= 12 && st.ticks <= 17);     // 2 s * 7.82 Hz = 15.6
        CHECK(st.wins >= st.ticks - 1);
        CHECK(std::fabs(st.measured_hz - 7.82) < 0.5);
    }
}

// ---------------------------------------------------------------------------
static void test_math_skill() {
    SECTION("Math skill - natural language arithmetic with inner thinking");
    Skill m = make_math_skill();
    CHECK(m.valid());

    struct Case { const char* q; double expect; };
    const Case cases[] = {
        { "what is 100*100 and add 3/4 or sum divided by 1.34%", 10000.75 / 0.0134 },
        { "calculate (12 + 8) * 3 minus 10 percent", 54.0 },
        { "2^10 plus square root of 144", 1036.0 },
        { "what is 20% of 50", 10.0 },
        { "100 minus 10%", 90.0 },
        { "1,000 divided by 8 then multiply it by 3 and subtract 5", 370.0 },
        { "3 x 4", 12.0 },
        { "how much is 7 times 6?", 42.0 },
        { "what is 2 plus 2 and then the result squared", 16.0 },
        { "-5 + 10", 5.0 },
    };
    for (const auto& c : cases) {
        SkillContext sc; sc.text = c.q; SkillResult r = m.run(sc);
        CHECK(r.handled);
        CHECK(!r.steps.empty());
        double v = 0.0;
        try { v = std::stod(r.answer); } catch (...) { v = -1e300; }
        if (std::fabs(v - c.expect) > 1e-6 * std::max(1.0, std::fabs(c.expect))) {
            std::cerr << "  math mismatch for \"" << c.q << "\": got " << r.answer << " expected " << c.expect << "\n";
            CHECK(false);
        }
    }
    { SkillContext sc; sc.text = cases[0].q; std::cout << "  " << m.run(sc).thinking << std::endl; }

    // prose with numbers is NOT a calculation
    auto run = [&](const char* q) { SkillContext sc; sc.text = q; return m.run(sc); };
    CHECK(!run("sensor: temperature spike detected at node 4 value 80C").handled);
    CHECK(!run("alert: memory usage exceeded 90 percent on node 2").handled);
    CHECK(!run("what is the temperature on node 3").handled);

    // math errors are reported, not hidden
    SkillResult z = run("5 / 0");
    CHECK(z.handled);
    CHECK(z.answer.find("division by zero") != std::string::npos);

    CHECK(std::fabs(evaluate_arithmetic("(1 + 2) * 3") - 9.0) < 1e-9);
    CHECK(std::fabs(evaluate_arithmetic("# * 2", 21.0) - 42.0) < 1e-9);
    bool threw = false;
    try { evaluate_arithmetic("1 +"); } catch (const std::exception&) { threw = true; }
    CHECK(threw);
}

// ---------------------------------------------------------------------------
static void test_ask_api() {
    SECTION("OpenRAI::ask() - text in, answer + thinking out");
    EngineConfig c = test_config();
    c.expert_count = 9;            // include E8:math
    c.max_concurrent = 3;
    OpenRAI engine(c);
    engine.start();

    auto f1 = engine.ask("what is 100*100 and add 3/4 or sum divided by 1.34%");
    CHECK(f1.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    Answer a1 = f1.get();
    CHECK(a1.solved);
    CHECK(a1.expert == "E8:math");
    CHECK(a1.answer.rfind("746324.62", 0) == 0);
    CHECK(a1.confidence > 0.9f);
    CHECK(a1.thinking.find("[PRIMARY]") != std::string::npos);
    CHECK(a1.thinking.find("[ARBITER]") != std::string::npos);
    // the math expert was chosen by its *model*, not by a keyword list
    CHECK(a1.thinking.find("E8:math") != std::string::npos);
    CHECK(a1.thinking.find("model-competence=") != std::string::npos);
    CHECK(a1.thinking.find("keyword-affinity") == std::string::npos);
    CHECK(a1.thinking.find("[E8:math] familiarity logp=") != std::string::npos);
    CHECK(a1.thinking.find("clause 3") != std::string::npos);
    CHECK(a1.tick >= 1);
    CHECK(a1.latency_ms > 0.0 && a1.latency_ms < 1000.0);
    std::cout << "  Q: " << a1.question << "\n  A: " << a1.answer << "  (" << a1.expert
              << ", " << a1.latency_ms << " ms)" << std::endl;

    // non-math input is answered by the semantic pipeline, not the math expert
    auto f2 = engine.ask("alert: memory usage exceeded 90 percent on node 2");
    CHECK(f2.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    Answer a2 = f2.get();
    CHECK(!a2.solved);
    CHECK(!a2.answer.empty());
    CHECK(!a2.expert.empty() && a2.expert != "-");
    std::cout << "  Q: " << a2.question << "\n  A: " << a2.answer << "  (" << a2.expert << ")" << std::endl;

    // two questions queued back to back are answered on consecutive ticks
    auto f3 = engine.ask("7 times 6");
    auto f4 = engine.ask("2^8");
    CHECK(f3.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    CHECK(f4.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    Answer a3 = f3.get(), a4 = f4.get();
    CHECK(a3.answer == "42");
    CHECK(a4.answer == "256");
    CHECK(a4.tick > a3.tick);

    // pending questions are released when the engine stops
    engine.stop();
    auto f5 = engine.ask("1+1");
    CHECK(f5.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
    CHECK(f5.get().answer.find("no answer") != std::string::npos);
}

// ---------------------------------------------------------------------------
static void test_gpu_selftest() {
    SECTION("GPU concurrency self-test API");
    auto rep = run_gpu_concurrency_selftest(NetArch::parse("32,256,128,4", "tiny"), 4, 4, 5);
    CHECK(rep.streams == 4);
    CHECK(rep.stream_info.size() == 4);
    CHECK(rep.serial_ms > 0.0 && rep.concurrent_ms > 0.0 && rep.threaded_ms > 0.0);
    CHECK(rep.max_result_diff < 1e-5);
    CHECK(!rep.verdict.empty());
    std::cout << rep.describe();
}

// ---------------------------------------------------------------------------
static void test_expert_models() {
    SECTION("Expert models: 3 test nodes, same 10k-word data, 200 iterations");
    TrainingDemoOptions opt;
    opt.nodes = 3;
    opt.iterations = 200;
    opt.words = 10000;
    auto rep = run_training_demo(opt);
    std::cout << rep.describe();
    CHECK(rep.passed);
    CHECK(rep.nodes.size() == 3);
    CHECK(rep.corpus_words >= 10000);
    for (const auto& n : rep.nodes) {
        CHECK(n.report.iterations == 200);
        CHECK(n.report.losses.size() == 200);
        CHECK(n.report.last_loss < 0.6f * n.report.first_loss);
        CHECK(n.in_domain.mean_logprob > n.foreign.mean_logprob + 2.f);
        CHECK(n.in_domain.coverage > 0.95f);
        CHECK(n.mlp_accuracy > 0.9f);
        CHECK(!n.samples.empty());
    }

    // save / load round trip keeps the knowledge
    auto corpus = corpus_texts(generate_domain_corpus("anomaly", 4000, 3));
    ExpertModel m("anomaly");
    m.build_vocab(corpus);
    auto r = m.train(corpus, 120, 64);
    CHECK(r.last_loss < r.first_loss);
    const std::string probe = "anomaly score 0.91 above threshold on node 4";
    const float before = m.score(probe).mean_logprob;
    CHECK(m.save("data/test_models/_roundtrip.model"));
    ExpertModel m2("copy");
    CHECK(m2.load("data/test_models/_roundtrip.model"));
    CHECK(m2.trained() && m2.train_iterations() == 120);
    CHECK(std::fabs(m2.score(probe).mean_logprob - before) < 1e-4f);
    CHECK(m2.generate("anomaly score", 6) == m.generate("anomaly score", 6));
    CHECK(!ExpertModel("x").score("anything").valid);   // untrained -> invalid

    // the generic tokenizer: digits and operators are tokens, no domain words
    auto w = ExpertModel::words("What is 12*7 + 3?");
    CHECK(w.size() == 9 && w[0] == "what" && w[2] == "1" && w[3] == "2" && w[4] == "*" && w[8] == "?");

    // MLP backprop: loss must fall on a separable problem
    GPUNeuralNet net(NetArch::parse("32,64,32,4", "tiny"), 1, true);
    CHECK(net.is_trainable());
    std::vector<float> in(16 * 32), tg(16 * 4, 0.f);
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    for (size_t i = 0; i < 16; ++i) {
        for (size_t j = 0; j < 32; ++j) in[i * 32 + j] = d(rng);
        tg[i * 4] = in[i * 32] > 0.f ? 1.f : 0.f;
        tg[i * 4 + 1] = tg[i * 4 + 2] = tg[i * 4 + 3] = 0.f;
    }
    const float l0 = net.train_batch(in.data(), tg.data(), 16, 0.05f);
    float ln = l0;
    for (int it = 0; it < 150; ++it) ln = net.train_batch(in.data(), tg.data(), 16, 0.05f);
    CHECK(ln < 0.7f * l0);
    CHECK(net.train_steps() == 151);
    GPUNeuralNet frozen(NetArch::parse("32,64,32,4", "tiny"), 1, false);
    CHECK(frozen.train_batch(in.data(), tg.data(), 16, 0.05f) < 0.f);   // not trainable
}

// ---------------------------------------------------------------------------
static void test_experience_store() {
    SECTION("ExperienceStore (SQLite learning store)");
    const std::string path = "data/test_store.db";
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(path + "-wal", ec);
    std::filesystem::remove(path + "-shm", ec);

    ExperienceStore store;
    CHECK(store.open(path));
    CHECK(store.count() == 0);

    Experience e;
    e.question = "what is 7 times 6";
    e.answer   = "42";
    e.immediate = "42";
    e.expert   = "E8:math";
    e.confidence = 0.95f;
    e.solved = true;
    e.embedding = std::vector<float>(32, 0.25f);
    const int64_t id = store.add(e);
    CHECK(id == 1);
    CHECK(store.count() == 1);

    Experience g;
    CHECK(store.get(id, g));
    CHECK(g.question == e.question && g.answer == "42" && g.expert == "E8:math");
    CHECK(g.embedding.size() == 32 && std::fabs(g.embedding[5] - 0.25f) < 1e-6f);
    CHECK(g.uses == 0 && g.reward == 0.f);
    CHECK(store.reward(id, 1.f));
    CHECK(store.touch(id));
    CHECK(store.get(id, g));
    CHECK(g.uses == 1 && std::fabs(g.reward - 1.f) < 1e-6f);
    CHECK(!store.reward(999, 1.f));
    CHECK(store.recent(5).size() == 1);

    NodeStats st; st.trust = 0.77f; st.wins = 9; st.fired = 20;
    CHECK(store.save_node_stats("E1:anomaly", st, 88));
    NodeStats back; int prio = 0;
    CHECK(store.load_node_stats("E1:anomaly", back, prio));
    CHECK(std::fabs(back.trust - 0.77f) < 1e-6f && back.wins == 9 && back.fired == 20 && prio == 88);
    CHECK(!store.load_node_stats("nope", back, prio));

    CHECK(store.save_goals({ {"stability", 0.6f}, {"memory", 0.4f} }));
    auto goals = store.load_goals();
    CHECK(goals.size() == 2);

    CHECK(store.save_graph({ {"a", "b", 3}, {"b", "c", 1} }));
    auto edges = store.load_graph();
    CHECK(edges.size() == 2);
    CHECK(store.set_kv("k", "v") );
    std::string v;
    CHECK(store.get_kv("k", v) && v == "v");
    CHECK(!store.get_kv("missing", v));

    // persistence across open/close
    store.close();
    ExperienceStore again;
    CHECK(again.open(path));
    CHECK(again.count() == 1);
    CHECK(again.get(1, g) && g.answer == "42");
}

// ---------------------------------------------------------------------------
static void test_vector_index() {
    SECTION("GPUVectorIndex + VectorMemory (GPU/CPU cosine search)");
    const size_t dim = 32, n = 2000;
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::vector<std::vector<float>> vecs(n, std::vector<float>(dim));
    for (auto& v : vecs) for (auto& x : v) x = d(rng);

    GPUVectorIndex idx(dim, 64, 60, "test");   // small initial capacity -> growth path
    for (const auto& v : vecs) idx.add(v.data(), v.size());
    CHECK(idx.size() == n);

    std::vector<float> q(dim);
    for (auto& x : q) x = d(rng);
    auto hits = idx.search(q.data(), dim, 5);
    CHECK(hits.size() == 5);
    // brute-force reference
    auto cos = [&](const std::vector<float>& a, const std::vector<float>& b) {
        double dot = 0, na = 0, nb = 0;
        for (size_t j = 0; j < dim; ++j) { dot += a[j] * b[j]; na += a[j] * a[j]; nb += b[j] * b[j]; }
        return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
    };
    size_t best = 0; float best_s = -2.f;
    for (size_t i = 0; i < n; ++i) { const float s = cos(vecs[i], q); if (s > best_s) { best_s = s; best = i; } }
    CHECK(hits.front().index == best);
    CHECK(std::fabs(hits.front().score - best_s) < 1e-4f);
    CHECK(hits[0].score >= hits[1].score && hits[1].score >= hits[2].score);

    // exact item is found with similarity 1
    auto self = idx.search(vecs[123].data(), dim, 1);
    CHECK(self.front().index == 123 && std::fabs(self.front().score - 1.f) < 1e-4f);

    // batch == single
    std::vector<float> qb(2 * dim);
    std::copy(q.begin(), q.end(), qb.begin());
    std::copy(vecs[123].begin(), vecs[123].end(), qb.begin() + dim);
    auto batch = idx.search_batch(qb.data(), 2, 3);
    CHECK(batch.size() == 2 && batch[0].front().index == best && batch[1].front().index == 123);

    // rebuild with fewer items
    idx.rebuild({ vecs[0], vecs[1] });
    CHECK(idx.size() == 2);
    CHECK(idx.search(vecs[1].data(), dim, 1).front().index == 1);
    CHECK(idx.searches() >= 4);

    // VectorMemory on top of the index: store, tagged search, dedupe, eviction
    Tokenizer tok; EmbeddingLayer emb(32);
    VectorMemory mem(32, 50, true);
    for (int i = 0; i < 40; ++i) {
        const std::string text = "memory item number " + std::to_string(i) + (i % 2 ? " alpha" : " beta");
        mem.store(text, emb.embed(tok.tokenize(text)), i % 2 ? "experience:" + std::to_string(i) : "input");
    }
    CHECK(mem.size() == 40);
    mem.store("memory item number 3 alpha", emb.embed(tok.tokenize("memory item number 3 alpha")), "experience:3");
    CHECK(mem.size() == 40);      // exact duplicate suppressed
    auto h = mem.retrieve_context("memory item number 3 alpha", tok, emb, 3);
    CHECK(!h.empty() && h.front().text == "memory item number 3 alpha" && h.front().similarity > 0.99f);
    auto tagged = mem.search_tagged(emb.embed(tok.tokenize("memory item number 4 beta")), "experience:", 3);
    CHECK(tagged.size() == 3);
    for (const auto& t : tagged) CHECK(t.tag.rfind("experience:", 0) == 0);
    for (int i = 40; i < 60; ++i) {
        const std::string text = "late item " + std::to_string(i);
        mem.store(text, emb.embed(tok.tokenize(text)), "input");
    }
    CHECK(mem.size() <= 50);      // capacity respected, index rebuilt
    CHECK(mem.retrieve_context("late item 59", tok, emb, 1).front().text == "late item 59");
    std::cout << "  index backend=" << (mem.gpu_index_active() ? "GPU" : "CPU")
              << " last_search_us=" << mem.last_search_us() << std::endl;
}

// ---------------------------------------------------------------------------
static void test_lm_adapter() {
    SECTION("LM adapter: JSON helpers, mock backend, unreachable http");
    CHECK(json_escape("a\"b\\c\n") == "a\\\"b\\\\c\\n");
    std::string out;
    const std::string j = "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"He said \\\"hi\\\"\\nline2 \\u00e9 \\ud83d\\ude00\"}}]}";
    CHECK(json_extract_string(j, "content", out));
    CHECK(out == std::string("He said \"hi\"\nline2 \xC3\xA9 \xF0\x9F\x98\x80"));
    CHECK(json_extract_string(j, "role", out) && out == "assistant");
    CHECK(!json_extract_string(j, "missing", out));
    CHECK(!json_extract_string("{\"content\": 5}", "content", out));   // not a string

    std::string prompt = LMAdapter::build_prompt("q?", { "E8:math (math): SOLVED -> 42" }, { "mem line" }, { "stability:0.5" }, 6000);
    CHECK(prompt.find("Expert node findings") != std::string::npos);
    CHECK(prompt.find("- E8:math") != std::string::npos);
    CHECK(prompt.find("User: q?") != std::string::npos);

    LMConfig mc; mc.backend = "mock";
    LMAdapter mock(mc);
    CHECK(mock.initialize() && mock.is_ready());
    auto r = mock.generate_with_context("what is 6*7", { "E8:math (math): SOLVED -> 42" }, {}, {});
    CHECK(r.ok && r.text.find("SOLVED -> 42") != std::string::npos);

    LMConfig hc; hc.backend = "http"; hc.url = "http://127.0.0.1:1"; hc.timeout_s = 2;
    LMAdapter http(hc);
    CHECK(!http.initialize());
    CHECK(!http.is_ready());

    LMConfig oc; oc.backend = "off";
    LMAdapter off(oc);
    CHECK(!off.initialize());
}

// ---------------------------------------------------------------------------
static void test_learning_loop() {
    SECTION("Learning loop: mock LM phrasing, recall, feedback, persistence");
    EngineConfig c = test_config();
    c.expert_count   = 9;
    c.max_concurrent = 3;
    c.llm_backend    = "mock";
    c.store_path     = "data/test_learn.db";
    c.memory_path    = "data/test_learn_memory.db";
    c.memory_capacity = 5000;
    std::error_code ec;
    std::filesystem::remove(c.store_path, ec);
    std::filesystem::remove(c.store_path + "-wal", ec);
    std::filesystem::remove(c.store_path + "-shm", ec);
    std::filesystem::remove(c.memory_path, ec);

    int64_t first_id = 0;
    {
        OpenRAI engine(c);
        int callbacks = 0;
        engine.set_output_callback([&](const std::string&) { ++callbacks; });
        engine.start();

        // 1. math question: solved by E8:math, phrased by the (mock) language expert
        auto f1 = engine.ask("what is 7 times 6");
        CHECK(f1.wait_for(std::chrono::seconds(8)) == std::future_status::ready);
        Answer a1 = f1.get();
        CHECK(a1.solved);
        CHECK(a1.language);
        CHECK(a1.immediate == "42");
        CHECK(a1.answer.find("[mock-lm]") == 0);
        CHECK(a1.answer.find("SOLVED -> 42") != std::string::npos);
        CHECK(a1.expert.find("E7:language") == 0);
        CHECK(a1.experience_id > 0);
        CHECK(!a1.recalled);
        first_id = a1.experience_id;
        std::cout << "  A1: " << a1.answer << "  (" << a1.expert << ", exp#" << a1.experience_id << ")" << std::endl;

        // 2. same question: recalled from the learned experience, no LM round trip
        auto f2 = engine.ask("what is 7 times 6");
        CHECK(f2.wait_for(std::chrono::seconds(8)) == std::future_status::ready);
        Answer a2 = f2.get();
        CHECK(a2.recalled);
        CHECK(a2.experience_id == first_id);
        CHECK(a2.answer == a1.answer);
        CHECK(!a2.language);
        CHECK(a2.latency_ms < 600.0);
        std::cout << "  A2: recalled #" << a2.experience_id << " in " << a2.latency_ms << " ms" << std::endl;

        // 3. free-text question: no solver -> LM phrases it; asked again -> memory expert recalls
        auto f3 = engine.ask("why did node 3 overheat yesterday?");
        CHECK(f3.wait_for(std::chrono::seconds(8)) == std::future_status::ready);
        Answer a3 = f3.get();
        CHECK(a3.language && !a3.solved);
        CHECK(a3.experience_id > first_id);
        auto f4 = engine.ask("why did node 3 overheat yesterday?");
        CHECK(f4.wait_for(std::chrono::seconds(8)) == std::future_status::ready);
        Answer a4 = f4.get();
        CHECK(a4.recalled && a4.experience_id == a3.experience_id);
        CHECK(a4.expert == "E2:memory");
        CHECK(a4.thinking.find("RECALL experience") != std::string::npos);
        std::cout << "  A4: " << a4.expert << " recalled #" << a4.experience_id << std::endl;

        // 4. punish it -> the next identical question is not recalled
        CHECK(engine.feedback(a3.experience_id, -1.f));
        CHECK(engine.last_experience_id() == a3.experience_id);
        auto f5 = engine.ask("why did node 3 overheat yesterday?");
        CHECK(f5.wait_for(std::chrono::seconds(8)) == std::future_status::ready);
        Answer a5 = f5.get();
        CHECK(!a5.recalled);
        CHECK(a5.language);
        CHECK(a5.experience_id > a3.experience_id);

        CHECK(engine.search_memory("node 3 overheat", 3).size() == 3);
        CHECK(engine.recent_experiences(10).size() >= 3);
        CHECK(callbacks == 5);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));   // stats are published per tick
        auto st = engine.stats();
        CHECK(st.experiences >= 3 && st.recalls >= 2 && st.language_answers >= 3);
        engine.stop();
    }
    // 5. persistence: a fresh engine recalls what the previous one learned
    {
        OpenRAI engine(c);
        engine.start();
        auto f = engine.ask("what is 7 times 6");
        CHECK(f.wait_for(std::chrono::seconds(8)) == std::future_status::ready);
        Answer a = f.get();
        CHECK(a.recalled && a.experience_id == first_id);
        engine.stop();
    }
}

// ---------------------------------------------------------------------------
// Opt-in (env OPENRAI_RUN_BIG_TEST=1): confirms that a math model trained
// on ~500k mixed symbolic/worded arithmetic lines for 10000 iterations can
// answer held-out questions through ExpertModel::generate() alone - no
// parser, no calculator.  Takes a few minutes, so it is not part of the
// default test run.
static void test_big_training_demo() {
    SECTION("Large-scale confirmation (500k lines/words, 10000 iterations) - opt-in");
    BigTrainOptions opt;
    opt.math_lines = 500000;
    opt.language_words = 500000;
    opt.iterations = 10000;
    opt.verbose_progress = false;
    auto rep = run_big_training_demo(opt);
    std::cout << rep.describe();
    CHECK(rep.passed);
    CHECK(rep.eval_total == 300);
    CHECK(static_cast<double>(rep.eval_correct) / rep.eval_total >= 0.6);
    CHECK(rep.math_report.last_loss < rep.math_report.first_loss);
    CHECK(rep.lang_report.last_loss < rep.lang_report.first_loss);
}

// ---------------------------------------------------------------------------
int main() {
    std::cout << "OpenRAI node scheduler tests (engine log -> openrai.log)" << std::endl;
    Logger::get().set_console(false);
    test_scheduler();
    test_stream_context();
    test_expert_node_lifecycle();
    test_primary_node();
    test_math_skill();
    test_engine_smoke();
    test_ask_api();
    test_gpu_selftest();
    test_expert_models();
    if (std::getenv("OPENRAI_RUN_BIG_TEST")) test_big_training_demo();
    else std::cout << "\n[SKIP] large-scale confirmation (set OPENRAI_RUN_BIG_TEST=1, ~3 min)" << std::endl;
    test_experience_store();
    test_vector_index();
    test_lm_adapter();
    test_learning_loop();

    std::cout << "\n" << (g_checks - g_failures) << "/" << g_checks << " checks passed" << std::endl;
    if (g_failures) {
        std::cout << "FAILED" << std::endl;
        return 1;
    }
    std::cout << "ALL TESTS PASSED" << std::endl;
    return 0;
}
