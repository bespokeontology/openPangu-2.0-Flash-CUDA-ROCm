// tokcli - model-agnostic tokenizer + CLI toolkit for the gfx906 engines.
//
// Nothing here is specific to a model: it reads whatever tokenizer.json it is
// handed. These are the pieces a decode engine needs and that are easy to get
// subtly wrong, collected so another engine can take them instead of rewriting:
//
//   * text -> ids and ids -> text, against the model's own tokenizer
//   * an id-range check (check), because a prompt file tokenized by the WRONG
//     tokenizer silently indexes past the end of the embedding table
//   * a chat-template hook, with chatml() as a worked renderer
//   * the host sampler: temperature / top-k / top-p / seeded, the same
//     arithmetic the Pangu CLI uses, so results compare across engines
//
// Build: g++ -O2 -std=c++17 hf_tokenizer.cpp tokcli.cpp -lpcre2-8 -o tokcli
#include "hf_tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

static std::string slurp(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(2); }
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static std::vector<std::int32_t> read_ids(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(2); }
    std::vector<std::int32_t> ids;
    std::int64_t v = 0;
    while (f.read(reinterpret_cast<char *>(&v), 8)) ids.push_back(static_cast<std::int32_t>(v));
    return ids;
}

static void write_ids(const std::vector<std::int32_t> &ids, const char *out) {
    std::ofstream o(out, std::ios::binary);
    if (!o) { std::fprintf(stderr, "cannot write %s\n", out); std::exit(2); }
    for (std::int32_t t : ids) { const std::int64_t v = t; o.write(reinterpret_cast<const char *>(&v), 8); }
    std::fprintf(stderr, "wrote %zu tokens -> %s\n", ids.size(), out);
}

// A worked chat renderer. Real templates differ per model (Qwen ships
// chat_template.jinja); this is the shape both engines' CLIs use.
static std::string chatml(const std::string &system, const std::string &user) {
    std::string s;
    s += "<|im_start|>system\n" + system + "<|im_end|>\n";
    s += "<|im_start|>user\n" + user + "<|im_end|>\n";
    s += "<|im_start|>assistant\n";
    return s;
}

// Host sampling over full logits. T<=0 is greedy. Deterministic for a given seed.
static int sample_topkp(const std::vector<float> &lg, int K, float T, float PP, unsigned seed) {
    const int n = static_cast<int>(lg.size());
    if (n <= 0) return -1;
    if (K < 1) K = 1;
    if (K > n) K = n;
    std::vector<int> idx(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) idx[static_cast<std::size_t>(i)] = i;
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                      [&](int a, int b) { return lg[static_cast<std::size_t>(a)] > lg[static_cast<std::size_t>(b)]; });
    if (T <= 0.f) return idx[0];
    const float mx = lg[static_cast<std::size_t>(idx[0])];
    std::vector<float> pr(static_cast<std::size_t>(K));
    double sum = 0;
    for (int i = 0; i < K; ++i) { pr[static_cast<std::size_t>(i)] = std::exp((lg[static_cast<std::size_t>(idx[static_cast<std::size_t>(i)])] - mx) / T); sum += pr[static_cast<std::size_t>(i)]; }
    for (int i = 0; i < K; ++i) pr[static_cast<std::size_t>(i)] = static_cast<float>(pr[static_cast<std::size_t>(i)] / sum);
    double acc = 0;
    int K2 = K;
    for (int i = 0; i < K; ++i) { acc += pr[static_cast<std::size_t>(i)]; if (acc >= PP) { K2 = i + 1; break; } }
    static std::mt19937 rng(seed);
    const double r = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    double c = 0;
    int pick = idx[0];
    for (int i = 0; i < K2; ++i) { c += pr[static_cast<std::size_t>(i)]; if (r <= c) { pick = idx[static_cast<std::size_t>(i)]; break; } }
    return pick;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage:\n"
            "  tokcli <tokenizer.json> info\n"
            "  tokcli <tokenizer.json> encode <text-in> <tokens-out.bin>\n"
            "  tokcli <tokenizer.json> chatml <system> <user> <tokens-out.bin>\n"
            "  tokcli <tokenizer.json> decode <tokens-in.bin>\n"
            "  tokcli <tokenizer.json> check  <tokens-in.bin>\n"
            "  tokcli <tokenizer.json> ids    < decimal ids on stdin\n"
            "  tokcli <tokenizer.json> sample <topk> <temp> <topp> <seed> < logits on stdin\n");
        return 2;
    }
    const std::string mode = argv[2];
    hf::Tokenizer tk(argv[1]);

    if (mode == "info") {
        const std::vector<std::int32_t> probe = tk.encode("The quick brown fox jumps over the lazy dog.");
        std::printf("round-trip: %s\n", tk.decode(probe) == "The quick brown fox jumps over the lazy dog." ? "ok" : "DIFFERS");
        return 0;
    }
    if (mode == "encode") {
        if (argc < 5) { std::fprintf(stderr, "encode needs <text-in> <tokens-out.bin>\n"); return 2; }
        write_ids(tk.encode(slurp(argv[3])), argv[4]);
        return 0;
    }
    if (mode == "chatml") {
        if (argc < 6) { std::fprintf(stderr, "chatml needs <system> <user> <tokens-out.bin>\n"); return 2; }
        write_ids(tk.encode(chatml(argv[3], argv[4])), argv[5]);
        return 0;
    }
    if (mode == "ids") {
        std::vector<std::int32_t> ids;
        long long v = 0;
        while (std::cin >> v) ids.push_back(static_cast<std::int32_t>(v));
        const std::string s = tk.decode(ids);
        std::fwrite(s.data(), 1, s.size(), stdout);
        return 0;
    }
    if (mode == "sample") {
        if (argc < 7) { std::fprintf(stderr, "sample needs <topk> <temp> <topp> <seed>\n"); return 2; }
        std::vector<float> lg;
        float v = 0;
        while (std::cin >> v) lg.push_back(v);
        const int pick = sample_topkp(lg, std::atoi(argv[3]), static_cast<float>(std::atof(argv[4])),
                                      static_cast<float>(std::atof(argv[5])),
                                      static_cast<unsigned>(std::atoi(argv[6])));
        std::printf("%d\n", pick);
        return 0;
    }
    if (argc < 4) { std::fprintf(stderr, "%s needs a file\n", mode.c_str()); return 2; }
    const std::vector<std::int32_t> ids = read_ids(argv[3]);
    if (mode == "decode") {
        const std::string s = tk.decode(ids);
        std::fwrite(s.data(), 1, s.size(), stdout);
        return 0;
    }
    if (mode == "check") {
        long long mn = 1LL << 62, mx = -1;
        for (std::int32_t t : ids) { if (t < mn) mn = t; if (t > mx) mx = t; }
        std::printf("tokens %zu min %lld max %lld\n", ids.size(), mn, mx);
        return 0;
    }
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
