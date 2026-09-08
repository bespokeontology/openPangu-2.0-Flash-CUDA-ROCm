#include "p92/model.h"
#include "p92/tokenizer.h"
#include "native_agent.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::int32_t kTextEnd = 148900;
constexpr std::int32_t kMessageEnd = 148902;
constexpr std::uint32_t kDefaultContext = 524288;
constexpr std::uint32_t kDefaultMaxNew = 8192;
constexpr std::uint32_t kDefaultIterations = 100;

volatile std::sig_atomic_t g_cancel = 0;

void on_signal(int) {
    g_cancel = 1;
}

std::string env_or(const char* name, std::string fallback) {
    if (const char* value = std::getenv(name); value && *value) return value;
    return fallback;
}

std::uint32_t parse_u32(const std::string& text, const char* label) {
    std::size_t consumed = 0;
    const unsigned long value = std::stoul(text, &consumed, 10);
    if (consumed != text.size() || value == 0 || value > 0xffffffffUL) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<std::uint32_t>(value);
}

struct Options final {
    std::string checkpoint = env_or(
        "P92_CHECKPOINT",
        "./openPangu-2.0-Flash");
    std::string artifact = env_or(
        "P92_NVFP4_ARTIFACT",
        "./openPangu-2.0-Flash-NVFP4-native");
    std::string cwd = std::filesystem::current_path().string();
    std::string system = env_or(
        "P92_AGENT_SYSTEM",
        "You are a rigorous local autonomous agent operating directly on the user's "
        "Linux computer. Use the native tools when they materially help. Terminal and "
        "filesystem actions are real, so inspect results, recover from errors, and do "
        "not claim work succeeded unless the tool result proves it. For ordinary "
        "questions that need no tool, answer directly.");
    std::uint32_t context = kDefaultContext;
    std::uint32_t max_new = kDefaultMaxNew;
    std::uint32_t max_iterations = kDefaultIterations;
    std::string initial_mode = env_or("P92_AGENT_MODE", "coding");
    bool thinking = false;
    bool speculative = true;
    bool verbose = false;
    bool protocol_selftest = false;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        const auto value = [&](const char* label) -> std::string {
            if (++index >= argc) throw std::invalid_argument(std::string(label) + " needs a value");
            return argv[index];
        };
        if (arg == "--checkpoint") options.checkpoint = value("--checkpoint");
        else if (arg == "--artifact") options.artifact = value("--artifact");
        else if (arg == "--cwd") options.cwd = value("--cwd");
        else if (arg == "--system") options.system = value("--system");
        else if (arg == "--context") options.context = parse_u32(value("--context"), "context");
        else if (arg == "--max-new") options.max_new = parse_u32(value("--max-new"), "max-new");
        else if (arg == "--iterations") {
            options.max_iterations = parse_u32(value("--iterations"), "iterations");
        } else if (arg == "--mode") options.initial_mode = value("--mode");
        else if (arg == "--thinking") options.thinking = true;
        else if (arg == "--verbose") options.verbose = true;
        else if (arg == "--no-speculative") options.speculative = false;
        else if (arg == "--selftest") options.protocol_selftest = true;
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "usage: p92_agent [--checkpoint DIR] [--artifact DIR] "
                   "[--cwd DIR] [--context N] "
                   "[--max-new N] [--iterations N] [--thinking] "
                   "[--mode coding|direct|research|writing] [--verbose] "
                   "[--no-speculative] "
                   "[--selftest]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (options.context > kDefaultContext) {
        throw std::invalid_argument("context cannot exceed 524288");
    }
    if (!std::filesystem::is_directory(options.checkpoint)) {
        throw std::invalid_argument("checkpoint directory not found: " + options.checkpoint);
    }
    if (!std::filesystem::is_directory(options.artifact)) {
        throw std::invalid_argument("artifact directory not found: " + options.artifact);
    }
    if (!std::filesystem::is_regular_file(
            std::filesystem::path(options.checkpoint) / "tokenizer.json")) {
        throw std::invalid_argument("tokenizer not found under checkpoint");
    }
    if (!std::filesystem::is_directory(options.cwd)) {
        throw std::invalid_argument("working directory not found: " + options.cwd);
    }
    if (options.initial_mode != "coding" && options.initial_mode != "direct" &&
        options.initial_mode != "research" &&
        options.initial_mode != "writing") {
        throw std::invalid_argument(
            "mode must be coding, direct, research, or writing");
    }
    return options;
}

std::string plain_system(std::string_view system) {
    std::string out = "<|pangu_text_start|><|message_start|>system\n";
    out.append(system);
    out += "<|message_end|>";
    return out;
}

std::string render_tool_definitions() {
    std::string out = "[";
    const std::size_t official = ddp_official_tool_count();
    const std::size_t extensions = ddp_host_extension_count();
    for (std::size_t index = 0; index < official + extensions; ++index) {
        const DdpToolDef* definition = index < official
            ? ddp_official_tool_at(index)
            : ddp_host_extension_at(index - official);
        if (index != 0) out += ',';
        out += "{\"type\":\"function\",\"function\":{\"name\":";
        out += nah_json_escape(definition->name);
        out += ",\"description\":";
        out += nah_json_escape(definition->description);
        out += ",\"parameters\":";
        out += definition->schema_json;
        out += "}}";
    }
    out += ']';
    return out;
}

std::string render_pangu_system_prompt(std::string_view additional) {
    std::string system = R"(你是一个能够调用外部工具解决问题的专家，你的目标是高效、准确、清晰地完成任务。
你需要根据用户的问题，决定是否需要使用工具来完成任务。如果需要，请以明确的格式调用工具；如果不需要，请直接回答。
你可以根据上下文决定是否继续调用工具或基于已有结果直接回答用户。如果工具调用已足够，请合理组织语言向用户汇报结论。在没有获得显式的调用结果之前，在调用工具的当轮回复之内严禁虚构或者假设一个工具调用结果来完成任务或者回答问题。也不应在没有返回工具调用信息的情况下，在调用工具的当轮假设或者明确声称工具执行成功。
你将在<tools></tools>标签对内获得每个工具的描述：
<tools>
)";
    system += render_tool_definitions();
    system += R"(
</tools>
对于每个函数调用，返回一个 JSON 对象，放在 <|tool_call_start|><|tool_call_end|> 标签对中，多个调用组成一个列表，其中每个函数包含函数名和对应函数的参数，格式如下：
<|tool_call_start|>
[{"name":"<函数名1>","arguments":{}},{"name":"<函数名2>","arguments":{}}]
<|tool_call_end|>
<工具使用原则>
1. 只有在所有必填参数(required字段中列出的)都具备有效值时，才能调用该函数
2. 如果缺少任何必填参数，必须向用户询问缺失的参数，而不是直接调用函数
3. 可选参数如果没有提供可以忽略或使用默认值
</工具使用原则>)";
    if (!additional.empty()) {
        system += "\n\n";
        system.append(additional);
    }
    system += "<|message_end|>";
    return "<|pangu_text_start|><|message_start|>system\n" + system;
}

std::string render_user_message(std::string_view content) {
    std::string out = "<|message_start|>user\n";
    out.append(content);
    out += "<|message_end|>";
    return out;
}

std::string render_assistant_prefix(bool thinking) {
    return thinking ? "<|message_start|>assistant\n<think>"
                    : "<|message_start|>assistant\n<think>\n\n</think>";
}

std::string render_tool_results(const std::vector<std::string>& payloads) {
    std::string out;
    for (const std::string& payload : payloads) {
        out += "<|message_start|>tool\n";
        out += payload;
        out += "<|message_end|>";
    }
    return out;
}

std::string trim_answer(std::string value) {
    if (const std::size_t thinking = value.rfind("</think>");
        thinking != std::string::npos) {
        value.erase(0, thinking + std::string_view("</think>").size());
    }
    while (!value.empty() &&
           (value.front() == ' ' || value.front() == '\t' ||
            value.front() == '\r' || value.front() == '\n')) {
        value.erase(value.begin());
    }
    while (!value.empty() &&
           (value.back() == ' ' || value.back() == '\t' ||
            value.back() == '\r' || value.back() == '\n')) {
        value.pop_back();
    }
    return value;
}

std::string role_system(std::string_view role) {
    if (role == "objective-seeker") {
        return "You are an objective research subagent. Use search, crawler, file, and "
               "terminal tools to complete the assigned task. Verify findings. When done, "
               "call info_seeker_objective_task_done with the evidence and key files.";
    }
    if (role == "subjective-seeker") {
        return "You are a research subagent gathering material for a writing task. Use the "
               "available tools and save useful material. When sufficient, call "
               "info_seeker_subjective_task_done.";
    }
    if (role == "writer") {
        return "You are a long-form writer subagent. Inspect the supplied files, use the "
               "classification, chapter-writing, and file tools, then call "
               "writer_subjective_task_done with the final artifact and summary.";
    }
    return "You are a rigorous local autonomous agent. Use tools until the task is done.";
}

std::string completion_nudge(std::string_view role) {
    if (role == "coding") {
        return
            "Three coding iterations remain. Stop broadening the task. Inspect the current "
            "diff and the latest build or test output, make only the fixes still required, "
            "run the most relevant final verification, then answer the user directly with "
            "the implemented result, evidence, and any real blocker. Do not call a planner, "
            "seeker, writer, or research completion tool.";
    }
    std::string tool = "the appropriate task completion tool";
    if (role == "objective-seeker") tool = "info_seeker_objective_task_done";
    else if (role == "subjective-seeker") tool = "info_seeker_subjective_task_done";
    else if (role == "writer") tool = "writer_subjective_task_done";
    else if (role == "planner-objective") tool = "planner_objective_task_done";
    else if (role == "planner-subjective") tool = "planner_subjective_task_done";
    return "Three iterations remain. Stop broadening the task, consolidate the evidence "
           "and saved files, and call " + tool +
           " as soon as the assigned task is complete.";
}

bool parse_pangu_completion(std::string_view raw,
                            native_agent::Completion* output,
                            std::string* error) {
    constexpr std::string_view open = "<|tool_call_start|>";
    constexpr std::string_view close = "<|tool_call_end|>";
    if (output == nullptr) {
        if (error) *error = "null completion output";
        return false;
    }
    output->text.clear();
    output->tool_calls.clear();
    const std::size_t begin = raw.find(open);
    if (begin == std::string_view::npos) {
        output->text = std::string(raw);
        return true;
    }
    const std::size_t end = raw.find(close, begin + open.size());
    if (end == std::string_view::npos) {
        if (error) *error = "unclosed Pangu tool call";
        return false;
    }
    output->text = std::string(raw.substr(0, begin));
    NahJson envelope;
    std::string parse_error;
    const std::string_view json = raw.substr(begin + open.size(), end - begin - open.size());
    if (!nah_json_parse(json.data(), json.size(), &envelope, &parse_error)) {
        if (error) *error = "invalid Pangu tool JSON: " + parse_error;
        return false;
    }
    std::vector<NahJson> calls;
    if (envelope.is_array()) calls = envelope.a;
    else if (envelope.is_object()) calls.push_back(envelope);
    else {
        if (error) *error = "Pangu tool envelope must be an object or array";
        return false;
    }
    for (const NahJson& item : calls) {
        const std::string name = item.get_string("name");
        const NahJson* arguments = item.get("arguments");
        if (name.empty() || arguments == nullptr || !arguments->is_object()) {
            if (error) *error = "Pangu tool call needs name and object arguments";
            return false;
        }
        native_agent::ToolCall call;
        call.name = name;
        call.arguments = *arguments;
        output->tool_calls.push_back(std::move(call));
    }
    if (output->tool_calls.empty()) {
        if (error) *error = "empty Pangu tool call";
        return false;
    }
    const std::string_view suffix = raw.substr(end + close.size());
    if (suffix.find_first_not_of(" \t\r\n") != std::string_view::npos) {
        if (error) *error = "non-whitespace suffix after Pangu tool call";
        return false;
    }
    return true;
}

void protocol_selftest() {
    const std::size_t tools = ddp_official_tool_count() + ddp_host_extension_count();
    if (tools != 31) throw std::runtime_error("shared tool registry count changed");
    const std::string system = render_pangu_system_prompt("SELFTEST");
    if (system.find("\"name\":\"think\"") == std::string::npos ||
        system.find("\"name\":\"consult_deepseek\"") == std::string::npos ||
        system.find("<|tool_call_start|>") == std::string::npos) {
        throw std::runtime_error("Pangu system/tool rendering is incomplete");
    }
    native_agent::Completion completion;
    std::string error;
    const std::string fixture =
        "</think><|tool_call_start|>[{\"name\":\"bash\",\"arguments\":"
        "{\"command\":\"printf p92-agent-ok\"}}]<|tool_call_end|>";
    if (!parse_pangu_completion(fixture, &completion, &error) ||
        completion.tool_calls.size() != 1 ||
        completion.tool_calls.front().name != "bash" ||
        completion.tool_calls.front().arguments.get_string("command") !=
            "printf p92-agent-ok") {
        throw std::runtime_error("Pangu tool parser selftest failed: " + error);
    }
    std::cout << "P92_AGENT_PROTOCOL_SELFTEST_OK tools=" << tools << '\n';
}

struct Generation final {
    std::string text;
    std::uint32_t boundary = 0;
    std::uint32_t generated_tokens = 0;
    double seconds = 0.0;
    bool reached_limit = false;
};

struct SessionResult final {
    bool success = false;
    bool terminal = false;
    std::string answer;
    std::string payload;
    std::string error;
    std::uint32_t iterations = 0;
};

enum class AgentMode {
    coding,
    direct,
    research,
    writing,
};

std::string_view mode_name(AgentMode mode) {
    switch (mode) {
    case AgentMode::coding: return "coding";
    case AgentMode::direct: return "direct";
    case AgentMode::research: return "research";
    case AgentMode::writing: return "writing";
    }
    return "coding";
}

std::string mode_instructions(AgentMode mode) {
    if (mode == AgentMode::coding) {
        return
            "You are the primary coding agent for the current repository. Work through the "
            "task directly: inspect the tree and local instructions, read the relevant code, "
            "make focused edits, build or run proportionate tests, inspect their actual "
            "stdout and stderr, and iterate until the requested result works. Use bash, file "
            "search/read/edit tools, and background-job controls as one continuous coding "
            "loop. Preserve unrelated user changes. Use consult_deepseek only when a focused "
            "second opinion would unblock a concrete technical problem. The planner, seeker, "
            "writer, and their completion tools belong to the optional research/writing "
            "workflows, not ordinary coding. When the coding task is complete, report the "
            "implemented result and tests concisely.";
    }
    if (mode == AgentMode::research) {
        return
            "You are the lead objective research planner. Decompose difficult questions into "
            "independent evidence-gathering assignments with "
            "assign_multi_objective_tasks_to_info_seeker. Continue searching, crawling, "
            "reading, comparing, and consulting until the evidence is sufficient; long "
            "investigations are expected. Track contradictions and source paths. Do not stop "
            "merely because one search succeeded. When the objective is genuinely complete, "
            "call planner_objective_task_done with the synthesis, evidence files, and final "
            "answer.";
    }
    if (mode == AgentMode::writing) {
        return
            "You are the lead long-form research and writing planner. Assign subjective "
            "research tasks, classify the resulting source files, assign the writing task, "
            "inspect the finished artifact, and call planner_subjective_task_done only after "
            "the requested deliverable is complete.";
    }
    return {};
}

class DeepSeekClient final {
public:
    DeepSeekClient()
        : key_(env_or("DEEPSEEK_API_KEY", "")),
          endpoint_(env_or("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions")),
          model_(env_or("DEEPSEEK_MODEL", "deepseek-v4-pro")) {
        while (!endpoint_.empty() && endpoint_.back() == '/') endpoint_.pop_back();
        if (!endpoint_.ends_with("/chat/completions")) {
            if (!endpoint_.ends_with("/v1")) endpoint_ += "/v1";
            endpoint_ += "/chat/completions";
        }
    }

    bool configured() const noexcept { return !key_.empty(); }
    const std::string& model() const noexcept { return model_; }

    std::string generate(std::string_view system,
                         std::string_view user,
                         std::uint32_t max_new_tokens) const {
        if (!configured()) {
            throw std::runtime_error(
                "DeepSeek route selected but DEEPSEEK_API_KEY is not set");
        }
        NahJson messages = NahJson::array();
        NahJson system_message = NahJson::object();
        system_message.set("role", NahJson::string("system"));
        system_message.set("content", NahJson::string(std::string(system)));
        messages.append(std::move(system_message));
        NahJson user_message = NahJson::object();
        user_message.set("role", NahJson::string("user"));
        user_message.set("content", NahJson::string(std::string(user)));
        messages.append(std::move(user_message));
        NahJson payload = NahJson::object();
        payload.set("model", NahJson::string(model_));
        payload.set("messages", std::move(messages));
        payload.set("max_tokens", NahJson::integer(max_new_tokens));
        payload.set("stream", NahJson::boolean(false));
        NahJson thinking = NahJson::object();
        thinking.set("type", NahJson::string("enabled"));
        payload.set("thinking", std::move(thinking));
        payload.set("reasoning_effort", NahJson::string("high"));
        const NahHttpResponse response = NahHttp::post_json(
            endpoint_, payload.dump(), 180000,
            {"Authorization: Bearer " + key_});
        if (!response.error.empty() && response.body.empty()) {
            throw std::runtime_error("DeepSeek request failed: " + response.error);
        }
        NahJson decoded;
        std::string parse_error;
        if (!nah_json_parse_text(response.body.c_str(), &decoded, &parse_error)) {
            throw std::runtime_error("DeepSeek returned invalid JSON: " + parse_error);
        }
        if (const NahJson* api_error = decoded.get("error")) {
            throw std::runtime_error("DeepSeek API error: " + api_error->dump());
        }
        const std::vector<NahJson>* choices = decoded.get_array("choices");
        if (!choices || choices->empty()) {
            throw std::runtime_error("DeepSeek response has no choices");
        }
        const NahJson* message = choices->front().get("message");
        if (!message || !message->is_object()) {
            throw std::runtime_error("DeepSeek response has no message");
        }
        std::string content = message->get_string("content");
        if (content.empty()) content = message->get_string("reasoning_content");
        if (content.empty()) {
            throw std::runtime_error("DeepSeek response message is empty");
        }
        return content;
    }

private:
    std::string key_;
    std::string endpoint_;
    std::string model_;
};

class NativeAgent final {
public:
    explicit NativeAgent(Options options)
        : options_(std::move(options)),
          tokenizer_(std::filesystem::path(options_.checkpoint) / "tokenizer.json"),
          model_(std::make_unique<p92::Model>(
              options_.checkpoint, options_.artifact, options_.context,
              p92::Model::WeightMode::fully_resident, options_.speculative)),
          host_(options_.cwd),
          max_new_(options_.max_new),
          max_iterations_(options_.max_iterations),
          thinking_(options_.thinking),
          verbose_(options_.verbose) {
        if (options_.initial_mode == "direct") mode_ = AgentMode::direct;
        else if (options_.initial_mode == "research") mode_ = AgentMode::research;
        else if (options_.initial_mode == "writing") mode_ = AgentMode::writing;
        if (mode_ != AgentMode::direct && max_iterations_ < 100) max_iterations_ = 100;
    }

    ~NativeAgent() { host_.shutdown(); }

    p92::Model& model() { return *model_; }
    NahHost& host() { return host_; }
    std::uint32_t context() const { return options_.context; }
    std::uint32_t max_new() const { return max_new_; }
    std::uint32_t max_iterations() const { return max_iterations_; }
    bool thinking() const { return thinking_; }
    bool speculative() const { return options_.speculative; }
    bool deepseek_configured() const { return deepseek_.configured(); }
    const std::string& deepseek_model() const { return deepseek_.model(); }
    bool first_turn() const { return first_turn_; }
    std::int32_t boundary() const { return boundary_; }
    AgentMode mode() const { return mode_; }

    void set_max_new(std::uint32_t value) { max_new_ = value; }
    void set_max_iterations(std::uint32_t value) { max_iterations_ = value; }
    void set_thinking(bool value) { thinking_ = value; }

    void set_mode(AgentMode mode) {
        mode_ = mode;
        if (mode_ != AgentMode::direct && max_iterations_ < 100) {
            max_iterations_ = 100;
        }
        reset_chat();
    }

    void reset_chat() {
        model_->reset();
        active_history_.clear();
        first_turn_ = true;
        boundary_ = kMessageEnd;
    }

    SessionResult user_turn(const std::string& user) {
        std::string system = options_.system;
        const std::string mode_system = mode_instructions(mode_);
        if (!mode_system.empty()) system += "\n\n" + mode_system;
        std::string suffix;
        if (first_turn_) {
            model_->reset();
            active_history_.clear();
            suffix = render_pangu_system_prompt(system) +
                     render_user_message(user) +
                     render_assistant_prefix(thinking_);
        } else {
            if (boundary_ == kTextEnd) {
                reset_chat();
                suffix = render_pangu_system_prompt(system) +
                         render_user_message(user) +
                         render_assistant_prefix(thinking_);
            } else {
                suffix = render_user_message(user) +
                         render_assistant_prefix(thinking_);
            }
        }
        const std::string_view loop_role = mode_ == AgentMode::coding
            ? std::string_view("coding")
            : (mode_ == AgentMode::research
                ? std::string_view("planner-objective")
                : (mode_ == AgentMode::writing
                    ? std::string_view("planner-subjective") : std::string_view{}));
        SessionResult result = run_loop(
            std::move(suffix), max_iterations_, true, loop_role);
        if (result.success) first_turn_ = false;
        return result;
    }

private:
    Generation generate(const std::string& suffix, std::uint32_t max_new) {
        const std::vector<std::int32_t> encoded = tokenizer_.encode(suffix);
        if (encoded.empty()) throw std::runtime_error("agent encoded an empty prompt");
        if (active_history_.size() + encoded.size() >= options_.context) {
            throw std::runtime_error("agent context is full");
        }
        const auto begin = std::chrono::steady_clock::now();
        std::int32_t prediction = model_->prefill(encoded);
        active_history_.insert(active_history_.end(), encoded.begin(), encoded.end());
        std::vector<std::int32_t> output;
        output.reserve(max_new);
        Generation result;
        while (result.generated_tokens < max_new) {
            if (prediction == kMessageEnd || prediction == kTextEnd) {
                static_cast<void>(model_->forward(prediction));
                active_history_.push_back(prediction);
                result.boundary = static_cast<std::uint32_t>(prediction);
                break;
            }
            if (options_.speculative && max_new - result.generated_tokens >= 4) {
                const p92::Model::SpeculativeResult step = model_->speculative_step(prediction);
                bool stopped = false;
                for (std::uint32_t index = 0; index < step.emitted_count; ++index) {
                    const std::int32_t token = step.emitted[index];
                    active_history_.push_back(token);
                    if (!stopped && (token == kMessageEnd || token == kTextEnd)) {
                        result.boundary = static_cast<std::uint32_t>(token);
                        stopped = true;
                    } else if (!stopped) {
                        output.push_back(token);
                        ++result.generated_tokens;
                    }
                }
                prediction = step.next_token;
                if (stopped) break;
                continue;
            }
            output.push_back(prediction);
            active_history_.push_back(prediction);
            ++result.generated_tokens;
            prediction = model_->forward(prediction);
        }
        if (result.boundary == 0) {
            result.reached_limit = true;
            static_cast<void>(model_->forward(kMessageEnd));
            active_history_.push_back(kMessageEnd);
        }
        result.seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();
        result.text = tokenizer_.decode(output);
        return result;
    }

    template <class Function>
    auto isolated(Function&& function) -> decltype(function()) {
        const std::string key = "p92-agent-state-" + std::to_string(++snapshot_id_);
        const std::vector<std::int32_t> parent_history = active_history_;
        const std::int32_t parent_boundary = boundary_;
        const auto restore_parent = [&]() {
            try {
                model_->restore_state(key);
            } catch (...) {
                model_->release_state(key);
                throw;
            }
            model_->release_state(key);
            active_history_ = parent_history;
            boundary_ = parent_boundary;
        };
        model_->capture_state(key);
        try {
            model_->reset();
            active_history_.clear();
            auto result = function();
            restore_parent();
            return result;
        } catch (...) {
            try {
                restore_parent();
            } catch (...) {
            }
            throw;
        }
    }

    std::string generate_text(std::string_view system,
                              std::string_view user,
                              std::uint32_t max_new) {
        return isolated([&]() {
            const std::string suffix = plain_system(system) +
                render_user_message(user) +
                render_assistant_prefix(false);
            const Generation result = generate(suffix, max_new);
            if (result.boundary == 0) {
                throw std::runtime_error("nested generation exhausted its token limit");
            }
            return trim_answer(result.text);
        });
    }

    native_agent::ChildResult run_child(std::string_view role,
                                         std::string_view task,
                                         std::uint32_t max_iterations) {
        return isolated([&]() {
            const std::string suffix =
                render_pangu_system_prompt(role_system(role)) +
                render_user_message(task) +
                render_assistant_prefix(false);
            const SessionResult result = run_loop(suffix, max_iterations, false, role);
            native_agent::ChildResult child;
            child.success = result.success;
            child.final_answer = result.answer;
            child.payload = result.payload;
            child.error = result.error;
            return child;
        });
    }

    native_agent::DispatchResult execute(const native_agent::ToolCall& call) {
        native_agent::DispatchResult result =
            native_agent::dispatch_host_tool(&host_, call,
                                              reinterpret_cast<volatile int*>(&g_cancel));
        if (result.kind != native_agent::DispatchKind::RequiresModel) return result;
        if (call.name == "consult_deepseek") {
            if (!deepseek_.configured()) {
                return {native_agent::DispatchKind::Error,
                        nah_json_error("consult_deepseek requires DEEPSEEK_API_KEY"), {}};
            }
            const std::uint32_t max_tokens = static_cast<std::uint32_t>(
                std::clamp<long long>(call.arguments.get_int("max_tokens", 4096),
                                      1, 32768));
            const std::string question = call.arguments.get_string("question");
            const std::string context = call.arguments.get_string("context");
            const std::string purpose = call.arguments.get_string(
                "purpose", "Provide a second opinion and better search strategy.");
            const std::string system =
                "You are an external research and debugging consultant. You do not have "
                "live web access in this call. Analyze the supplied context, identify likely "
                "mistakes or missing angles, and give concrete alternative search queries, "
                "source targets, or next steps. Clearly separate known facts from hypotheses.";
            const std::string user = "PURPOSE:\n" + purpose + "\n\nQUESTION:\n" +
                                     question + "\n\nCONTEXT:\n" + context;
            NahJson data = NahJson::object();
            data.set("model", NahJson::string(deepseek_.model()));
            data.set("answer", NahJson::string(
                trim_answer(deepseek_.generate(system, user, max_tokens))));
            return {native_agent::DispatchKind::Complete,
                    nah_json_result(true, data, "", NahJson::object()), {}};
        }
        return native_agent::dispatch_model_tool(
            &host_, call,
            [&](std::string_view system, std::string_view user, std::uint32_t max_new) {
                return generate_text(system, user, max_new);
            },
            [&](std::string_view role, std::string_view task, std::uint32_t iterations) {
                return run_child(role, task, iterations);
            },
            reinterpret_cast<volatile int*>(&g_cancel));
    }

    SessionResult run_loop(std::string suffix,
                           std::uint32_t max_iterations,
                           bool announce,
                           std::string_view role = {}) {
        SessionResult session;
        for (std::uint32_t iteration = 1; iteration <= max_iterations; ++iteration) {
            if (g_cancel != 0) {
                session.error = "cancelled";
                return session;
            }
            if (!role.empty() && max_iterations > 3 &&
                iteration == max_iterations - 3) {
                const std::string assistant = render_assistant_prefix(false);
                if (suffix.ends_with(assistant)) suffix.resize(suffix.size() - assistant.size());
                suffix += render_user_message(completion_nudge(role));
                suffix += assistant;
            }
            if (verbose_) std::cerr << "[p92-agent] model step " << iteration << '\n';
            const Generation generated = generate(suffix, max_new_);
            session.iterations = iteration;
            if (verbose_) {
                const double rate = generated.seconds > 0.0
                    ? static_cast<double>(generated.generated_tokens) / generated.seconds : 0.0;
                std::cerr << "[p92-agent] tokens=" << generated.generated_tokens
                          << " wall=" << generated.seconds
                          << " tok_s=" << rate
                          << " position=" << model_->position() << '\n';
            }
            if (generated.reached_limit) {
                session.error = "model output hit --max-new before <|message_end|>";
                return session;
            }
            boundary_ = generated.boundary;
            native_agent::Completion completion;
            std::string parse_error;
            if (!parse_pangu_completion(generated.text, &completion, &parse_error)) {
                session.error = "malformed tool envelope: " + parse_error;
                return session;
            }
            if (completion.tool_calls.empty()) {
                session.success = true;
                session.answer = trim_answer(completion.text);
                return session;
            }
            if (announce && !completion.text.empty()) {
                std::cout << completion.text << '\n';
            }
            std::vector<std::string> payloads;
            payloads.reserve(completion.tool_calls.size());
            for (const native_agent::ToolCall& call : completion.tool_calls) {
                if (announce) std::cerr << "[tool] " << call.name << '\n';
                native_agent::DispatchResult dispatched = execute(call);
                payloads.push_back(dispatched.payload);
                if (dispatched.kind == native_agent::DispatchKind::Terminal) {
                    session.success = true;
                    session.terminal = true;
                    session.answer = dispatched.final_answer;
                    session.payload = dispatched.payload;
                    return session;
                }
            }
            if (boundary_ != kMessageEnd) {
                session.error = "tool response ended on EOS";
                return session;
            }
            suffix = render_tool_results(payloads) + render_assistant_prefix(false);
        }
        session.error = "agent reached its iteration limit";
        return session;
    }

    Options options_;
    p92::Tokenizer tokenizer_;
    std::unique_ptr<p92::Model> model_;
    NahHost host_;
    std::uint32_t max_new_;
    std::uint32_t max_iterations_;
    bool thinking_;
    bool verbose_;
    AgentMode mode_ = AgentMode::coding;
    DeepSeekClient deepseek_;
    bool first_turn_ = true;
    std::int32_t boundary_ = kMessageEnd;
    std::vector<std::int32_t> active_history_;
    std::uint64_t snapshot_id_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
        Options options = parse_options(argc, argv);
        if (options.protocol_selftest) {
            protocol_selftest();
            return 0;
        }
        const auto load_begin = std::chrono::steady_clock::now();
        NativeAgent agent(std::move(options));
        const auto load_end = std::chrono::steady_clock::now();
        const double resident_gib =
            (agent.model().projection_bytes() + agent.model().auxiliary_bytes()) /
            (1024.0 * 1024.0 * 1024.0);
        std::cout << "Native openPangu Flash-92 agent ready in "
                  << std::chrono::duration<double>(load_end - load_begin).count()
                  << "s; resident=" << resident_gib
                  << " GiB; context=" << agent.context()
                  << "; tools=" << (ddp_official_tool_count() + ddp_host_extension_count())
                  << "; speculative=" << (agent.speculative() ? "on" : "off")
                  << ".\n";
        std::cout << "Commands: /reset, /mode coding|direct|research|writing, "
                     "/thinking on|off, /max N, /iterations N, /cwd PATH, "
                     "/tools, /status, /quit\n";

        std::string line;
        while (true) {
            g_cancel = 0;
            std::cout << "\nYou> " << std::flush;
            if (!std::getline(std::cin, line)) break;
            if (line.empty()) continue;
            if (line == "/quit" || line == "/exit") break;
            if (line == "/reset") {
                agent.reset_chat();
                std::cout << "Conversation reset.\n";
                continue;
            }
            if (line == "/tools") {
                std::cout << "Tools (" << ddp_official_tool_count() + ddp_host_extension_count()
                          << "):";
                for (std::size_t index = 0; index < ddp_official_tool_count(); ++index) {
                    std::cout << ' ' << ddp_official_tool_at(index)->name;
                }
                for (std::size_t index = 0; index < ddp_host_extension_count(); ++index) {
                    std::cout << ' ' << ddp_host_extension_at(index)->name;
                }
                std::cout << '\n';
                continue;
            }
            if (line == "/status") {
                std::cout << "position=" << agent.model().position()
                          << " context=" << agent.context()
                          << " max_new=" << agent.max_new()
                          << " max_iterations=" << agent.max_iterations()
                          << " mode=" << mode_name(agent.mode())
                          << " thinking=" << (agent.thinking() ? "on" : "off")
                          << " deepseek="
                          << (agent.deepseek_configured() ? agent.deepseek_model() : "not-configured")
                          << " web_search="
                          << (std::getenv("SERPER_API_KEY") ? "serper->ddg->wikipedia"
                                                            : "ddg->wikipedia")
                          << " cwd=" << agent.host().fs.cwd() << '\n';
                continue;
            }
            if (line.starts_with("/mode ")) {
                const std::string value = line.substr(6);
                if (value == "coding") agent.set_mode(AgentMode::coding);
                else if (value == "direct") agent.set_mode(AgentMode::direct);
                else if (value == "research") agent.set_mode(AgentMode::research);
                else if (value == "writing") agent.set_mode(AgentMode::writing);
                else {
                    std::cout << "Use /mode coding, /mode direct, /mode research, "
                                 "or /mode writing.\n";
                    continue;
                }
                std::cout << "mode=" << mode_name(agent.mode())
                          << " max_iterations=" << agent.max_iterations()
                          << "; conversation reset.\n";
                continue;
            }
            if (line.starts_with("/thinking ")) {
                const std::string value = line.substr(10);
                if (value != "on" && value != "off") {
                    std::cout << "Use /thinking on or /thinking off.\n";
                } else {
                    agent.set_thinking(value == "on");
                    std::cout << "Thinking " << value << ".\n";
                }
                continue;
            }
            if (line.starts_with("/max ")) {
                agent.set_max_new(parse_u32(line.substr(5), "max-new"));
                std::cout << "max_new=" << agent.max_new() << '\n';
                continue;
            }
            if (line.starts_with("/iterations ")) {
                agent.set_max_iterations(parse_u32(line.substr(12), "iterations"));
                std::cout << "max_iterations=" << agent.max_iterations() << '\n';
                continue;
            }
            if (line.starts_with("/cwd ")) {
                const std::filesystem::path next = std::filesystem::absolute(line.substr(5));
                if (!std::filesystem::is_directory(next)) {
                    std::cout << "Directory not found: " << next << '\n';
                } else {
                    agent.host().fs.set_cwd(next.string());
                    std::cout << "cwd=" << next << '\n';
                }
                continue;
            }

            const SessionResult result = agent.user_turn(line);
            if (result.success) {
                std::cout << "Flash-92> " << result.answer << '\n';
            } else {
                std::cout << "Flash-92 agent error: " << result.error << '\n';
            }
        }
        std::cout << "P92_AGENT_SHUTDOWN\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "P92_AGENT_ERROR " << error.what() << '\n';
        return 1;
    }
}
