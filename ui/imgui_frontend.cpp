// -----------------------------------------------------------------------
// imgui_frontend.cpp
//   Dear ImGui text front-end for OpenRAI: type a question, get the answer
//   of the winning expert plus its inner thinking, watch the nodes live.
//
//   Build (needs a Dear ImGui checkout + GLFW + OpenGL):
//     cmake -DOPENRAI_IMGUI_DIR=/path/to/imgui ..
//     cmake --build . --target openrai_gui
//
//   The engine runs on its own thread at 7.82 Hz; the GUI only calls
//   OpenRAI::ask() (returns a std::future) and OpenRAI::stats().
//   NOTE: this file is not compiled in the default build; it has been
//   written against the stable Dear ImGui 1.8x/1.9x + GLFW/OpenGL3 API
//   but has not been built in the environment this was developed in.
// -----------------------------------------------------------------------

#include <openrai/core.h>
#include <openrai/config.h>
#include <openrai/log.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <future>
#include <memory>
#include <string>
#include <vector>

#ifdef OPENRAI_CUDA_ENABLED
static constexpr bool kCudaCompiled = true;
#else
static constexpr bool kCudaCompiled = false;
#endif

namespace {

struct Exchange {
    std::string question;
    std::future<openrai::Answer> future;
    bool   ready = false;
    openrai::Answer answer;
    bool   show_thinking = false;
};

void glfw_error(int code, const char* desc) {
    std::fprintf(stderr, "GLFW error %d: %s\n", code, desc);
}

} // namespace

int main(int, char**) {
    // Engine log stays in openrai.log; the GUI shows answers.
    openrai::Logger::get().set_console(false);

    const openrai::EngineConfig cfg = openrai::EngineConfig::from_env(kCudaCompiled);
    auto engine = std::make_unique<openrai::OpenRAI>(cfg);
    engine->start();

    glfwSetErrorCallback(glfw_error);
    if (!glfwInit()) return 1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    GLFWwindow* window = glfwCreateWindow(1280, 800, "OpenRAI - primary + expert nodes", nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    static char input_buf[4096] = "what is 100*100 and add 3/4 or sum divided by 1.34%";
    std::deque<Exchange> history;
    bool auto_scroll = true;
    bool focus_input = true;

    auto submit = [&]() {
        const std::string q(input_buf);
        if (q.empty()) return;
        Exchange ex;
        ex.question = q;
        ex.future   = engine->ask(q);
        history.push_back(std::move(ex));
        if (history.size() > 200) history.pop_front();
        input_buf[0] = '\0';
        focus_input = true;
    };

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Collect finished answers
        for (auto& ex : history) {
            if (ex.ready) continue;
            if (ex.future.valid() &&
                ex.future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                ex.answer = ex.future.get();
                ex.ready  = true;
            }
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        const ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->WorkPos);
        ImGui::SetNextWindowSize(vp->WorkSize);
        ImGui::Begin("OpenRAI", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

        // ---- status line -------------------------------------------------------
        const auto st = engine->stats();
        ImGui::Text("tick=%llu  %.2f Hz (target %.2f)  busy=%.1f ms  overruns=%llu  expert_wins=%llu  fallbacks=%llu  backend=%s%s",
                    static_cast<unsigned long long>(st.ticks), st.measured_hz, cfg.tick_hz, st.avg_busy_ms,
                    static_cast<unsigned long long>(st.overruns),
                    static_cast<unsigned long long>(st.wins),
                    static_cast<unsigned long long>(st.primary_fallbacks),
                    st.backend.c_str(), st.cuda_active ? " (streams active)" : "");
        ImGui::Text("memory=%zu items (%s index)  learned=%lld  recalls=%llu  LM=%s  lm_answers=%llu",
                    st.vector_memory, st.gpu_memory_index ? "GPU" : "CPU",
                    static_cast<long long>(st.experiences),
                    static_cast<unsigned long long>(st.recalls),
                    st.llm.c_str(),
                    static_cast<unsigned long long>(st.language_answers));
        ImGui::Separator();

        const float footer = ImGui::GetFrameHeightWithSpacing() * 4.0f;
        const float left_w = ImGui::GetContentRegionAvail().x * 0.66f;

        // ---- conversation --------------------------------------------------------
        ImGui::BeginChild("conversation", ImVec2(left_w, -footer), true);
        for (size_t i = 0; i < history.size(); ++i) {
            Exchange& ex = history[i];
            ImGui::PushID(static_cast<int>(i));
            ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "> %s", ex.question.c_str());
            if (!ex.ready) {
                ImGui::TextDisabled("   thinking... (waiting for the next tick)");
            } else {
                const auto& a = ex.answer;
                ImGui::TextColored(a.solved ? ImVec4(0.5f, 1.0f, 0.5f, 1.0f) : ImVec4(1.0f, 1.0f, 1.0f, 1.0f),
                                   "   %s", a.answer.c_str());
                ImGui::TextDisabled("   expert=%s%s%s%s  confidence=%.2f  utility=%.2f  tick=%llu  %.0f ms  exp#%lld",
                                    a.expert.c_str(), a.solved ? " (solved)" : "",
                                    a.recalled ? " (recalled)" : "", a.language ? " (LM)" : "",
                                    a.confidence, a.utility,
                                    static_cast<unsigned long long>(a.tick), a.latency_ms,
                                    static_cast<long long>(a.experience_id));
                ImGui::SameLine();
                if (ImGui::SmallButton("good")) engine->feedback(a.experience_id, 1.f);
                ImGui::SameLine();
                if (ImGui::SmallButton("bad"))  engine->feedback(a.experience_id, -1.f);
                ImGui::SameLine();
                if (ImGui::SmallButton(ex.show_thinking ? "hide thinking" : "show thinking"))
                    ex.show_thinking = !ex.show_thinking;
                if (ex.show_thinking) {
                    ImGui::Indent();
                    ImGui::PushTextWrapPos(0.0f);
                    ImGui::TextUnformatted(a.thinking.c_str());
                    ImGui::PopTextWrapPos();
                    ImGui::Unindent();
                }
            }
            ImGui::Spacing();
            ImGui::PopID();
        }
        if (auto_scroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 40.0f)
            ImGui::SetScrollHereY(1.0f);
        ImGui::EndChild();

        // ---- node panel ----------------------------------------------------------
        ImGui::SameLine();
        ImGui::BeginChild("nodes", ImVec2(0, -footer), true);
        ImGui::TextUnformatted("Expert nodes (primary decides who fires)");
        ImGui::Separator();
        for (const auto& line : st.node_lines) {
            ImVec4 col(0.85f, 0.85f, 0.85f, 1.0f);
            if (line.find("state=RUNNING")   != std::string::npos) col = ImVec4(0.4f, 1.0f, 0.4f, 1.0f);
            if (line.find("state=SUSPENDED") != std::string::npos) col = ImVec4(1.0f, 0.8f, 0.3f, 1.0f);
            if (line.find("state=KILLED")    != std::string::npos) col = ImVec4(1.0f, 0.4f, 0.4f, 1.0f);
            if (line.find("state=STOPPED")   != std::string::npos) col = ImVec4(0.6f, 0.6f, 0.6f, 1.0f);
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(col, "%s", line.c_str());
            ImGui::PopTextWrapPos();
        }
        ImGui::Separator();
        bool echo = openrai::Logger::get().console();
        if (ImGui::Checkbox("echo engine log to terminal", &echo)) openrai::Logger::get().set_console(echo);
        ImGui::Checkbox("auto-scroll", &auto_scroll);
        ImGui::EndChild();

        // ---- input ---------------------------------------------------------------------
        ImGui::Separator();
        ImGui::TextUnformatted("Ask (Enter to send):");
        if (focus_input) { ImGui::SetKeyboardFocusHere(); focus_input = false; }
        ImGui::PushItemWidth(-90.0f);
        const bool entered = ImGui::InputText("##ask", input_buf, sizeof(input_buf),
                                              ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::PopItemWidth();
        ImGui::SameLine();
        if (ImGui::Button("Ask", ImVec2(80, 0)) || entered) submit();

        ImGui::End();

        ImGui::Render();
        int w = 0, h = 0;
        glfwGetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    engine->stop();
    engine.reset();
    return 0;
}
