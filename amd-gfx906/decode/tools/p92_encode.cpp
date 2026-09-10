// p92_encode.cpp - native CPU driver for the model's own tokenizer.
// Encodes text into the engine's int64 token format, decodes it back, reports the
// id range so an out-of-vocabulary corpus cannot slip through, and can wrap text
// in the model's own chat template.
#include "p92/tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

static std::string slurp(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(2); }
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static void write_ids(const std::vector<std::int32_t> &ids, const char *out) {
    std::ofstream o(out, std::ios::binary);
    if (!o) { std::fprintf(stderr, "cannot write %s\n", out); std::exit(2); }
    long long mx = -1;
    for (std::int32_t t : ids) {
        if (t > mx) mx = t;
        const std::int64_t v = t;
        o.write(reinterpret_cast<const char *>(&v), 8);
    }
    std::fprintf(stderr, "encoded %zu tokens, max id %lld -> %s\n", ids.size(), mx, out);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage:\n"
            "  p92_encode <tokenizer.json> encode    <text-in> <tokens-out.bin>\n"
            "  p92_encode <tokenizer.json> chat      <text-in> <tokens-out.bin>  (wrap in chat template)\n"
            "  p92_encode <tokenizer.json> decode    <tokens-in.bin>\n"
            "  p92_encode <tokenizer.json> check     <tokens-in.bin>\n"
            "  p92_encode <tokenizer.json> ids       < <decimal ids on stdin>\n");
        return 2;
    }
    const std::string mode = argv[2];
    p92::Tokenizer tk(argv[1]);

    if (mode == "ids") {
        std::vector<std::int32_t> ids;
        long long v;
        while (std::cin >> v) ids.push_back(static_cast<std::int32_t>(v));
        const std::string s = tk.decode(ids);
        std::fwrite(s.data(), 1, s.size(), stdout);
        return 0;
    }

    if (mode == "encode" || mode == "chat") {
        if (argc < 5) { std::fprintf(stderr, "%s needs <text-in> <tokens-out.bin>\n", mode.c_str()); return 2; }
        std::string text = slurp(argv[3]);
        if (mode == "chat") text = p92::Tokenizer::chat_prompt(text);
        write_ids(tk.encode(text), argv[4]);
        return 0;
    }
    std::ifstream f(argv[3], std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[3]); return 2; }
    std::vector<std::int32_t> ids;
    std::int64_t v;
    while (f.read(reinterpret_cast<char *>(&v), 8)) ids.push_back(static_cast<std::int32_t>(v));

    if (mode == "decode") { const std::string s = tk.decode(ids); std::fwrite(s.data(), 1, s.size(), stdout); return 0; }
    if (mode == "check") {
        long long mn = 1LL << 62, mx = -1;
        for (std::int32_t t : ids) { if (t < mn) mn = t; if (t > mx) mx = t; }
        std::printf("tokens %zu min %lld max %lld\n", ids.size(), mn, mx);
        return 0;
    }
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
