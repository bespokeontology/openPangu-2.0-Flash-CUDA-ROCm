#include "p92amd/p92_arena.h"
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

#define NEMBD 2560
#define NFF   1024
#define GU_W  ((size_t)NFF   * (NEMBD/2))
#define GU_S  ((size_t)NFF   * (NEMBD/16))
#define DN_W  ((size_t)NEMBD * (NFF/2))
#define DN_S  ((size_t)NEMBD * (NFF/16))

static int slurp(int fd, void *dst_dev, size_t n, off_t off, std::vector<char> &stage) {
    const size_t CH = 64u << 20;
    if (stage.size() < CH) stage.resize(CH);
    size_t done = 0;
    while (done < n) {
        const size_t want = n - done < CH ? n - done : CH;
        size_t got = 0;
        while (got < want) {
            ssize_t k = pread(fd, stage.data() + got, want - got, off + done + got);
            if (k <= 0) return -1;
            got += k;
        }
        if (hipMemcpy((char *)dst_dev + done, stage.data(), want, hipMemcpyHostToDevice) != hipSuccess)
            return -1;
        done += want;
    }
    return 0;
}

int p92_arena_load(const char *path, P92Arena *a) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("arena open"); return -1; }
    if (pread(fd, &a->h, sizeof a->h, 0) != (ssize_t)sizeof a->h) { close(fd); return -1; }
    if (memcmp(a->h.magic, "P92AREN", 7)) { fprintf(stderr, "arena: bad magic\n"); close(fd); return -1; }
    a->slots = (int)(a->h.moe_layers * a->h.experts);
    const size_t n = (size_t)a->slots;
    struct { uint8_t **p; size_t bytes; uint64_t off; } pl[6] = {
        {&a->gw, n * GU_W, a->h.off_gw}, {&a->uw, n * GU_W, a->h.off_uw},
        {&a->dw, n * DN_W, a->h.off_dw}, {&a->gs, n * GU_S, a->h.off_gs},
        {&a->us, n * GU_S, a->h.off_us}, {&a->ds, n * DN_S, a->h.off_ds}};
    std::vector<char> stage;
    for (int i = 0; i < 6; i++) {
        if (hipMalloc(pl[i].p, pl[i].bytes) != hipSuccess) {
            fprintf(stderr, "arena: hipMalloc %.2f GB failed\n", pl[i].bytes / 1e9);
            close(fd); return -1; }
        if (slurp(fd, *pl[i].p, pl[i].bytes, (off_t)pl[i].off, stage)) { close(fd); return -1; }
    }
    if (hipMalloc(&a->s2, n * 3 * sizeof(float)) != hipSuccess) { close(fd); return -1; }
    if (slurp(fd, a->s2, n * 3 * sizeof(float), (off_t)a->h.off_s2, stage)) { close(fd); return -1; }
    close(fd);
    return 0;
}

void p92_arena_free(P92Arena *a) {
    hipFree(a->gw); hipFree(a->uw); hipFree(a->dw);
    hipFree(a->gs); hipFree(a->us); hipFree(a->ds); hipFree(a->s2);
    memset(a, 0, sizeof *a);
}
