#include "p92amd/p92_pf_drive.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

int pf_timers_init(PfTimers *t, int device, int enabled) {
    memset(t, 0, sizeof(*t));
    t->device = device; t->enabled = enabled;
    if (!enabled) return 0;
    if (hipSetDevice(device) != hipSuccess) return 1;
    for (int i = 0; i < PF_TM_SLOTS; i++) {
        if (hipEventCreate(&t->eb[i]) != hipSuccess) return 1;
        if (hipEventCreate(&t->ee[i]) != hipSuccess) return 1;
    }
    return 0;
}
void pf_timers_free(PfTimers *t) {
    if (!t->enabled) return;
    hipSetDevice(t->device);
    for (int i = 0; i < PF_TM_SLOTS; i++) { hipEventDestroy(t->eb[i]); hipEventDestroy(t->ee[i]); }
}
void pf_stage_accumulate(PfTimers *t) {
    if (!t->enabled) return;
    hipSetDevice(t->device);
    for (int i = 0; i < t->nslot; i++) {
        float ms = 0.f;
        if (hipEventElapsedTime(&ms, t->eb[i], t->ee[i]) == hipSuccess) t->ms[t->est[i]] += ms;
    }
    t->nslot = 0;
}

int pf_pipeline_init(PfPipeline *p, const PfConfig *cfg, int prompt_tokens) {
    p->cfg = *cfg;
    p->prompt_tokens = prompt_tokens;
    p->nchunks = (prompt_tokens + cfg->chunk - 1) / cfg->chunk;
    { const char *e;
      p->sens_stage = (e = getenv("PF_SENS_STAGE")) ? atoi(e) : -1;
      p->sens_reps  = (e = getenv("PF_SENS_REPS"))  ? atoi(e) : 1; }
    // 12/12/11/11, the decode ownership. Prefill inherits it because the arena
    // is placed that way and 50.02 GB does not fit any other split.
    const int per[P92_P2P_NGPU] = {12, 12, 11, 11};
    int at = 0;
    for (int d = 0; d < P92_P2P_NGPU; d++) {
        p->layer_begin[d] = at; at += per[d]; p->layer_end[d] = at;
    }
    // the landing buffer carries one chunk of four-stream state
    const size_t bytes = (size_t)cfg->chunk * 4 * 2560 * 2;
    if (p92_p2p_init(&p->p2p, bytes)) {
        fprintf(stderr, "pf_pipeline_init: peer transport refused to enable\n");
        return -1;
    }
    for (int d = 0; d < P92_P2P_NGPU; d++)
        if (pf_timers_init(&p->timers[d], d, cfg->verbose)) return -1;
    return 0;
}

void pf_pipeline_free(PfPipeline *p) {
    for (int d = 0; d < P92_P2P_NGPU; d++) pf_timers_free(&p->timers[d]);
    p92_p2p_free(&p->p2p);
}

void pf_report(const PfPipeline *p, const PfResult *r) {
    printf("\nprompt %d tokens, chunk %d, %d chunks, P%d/D4, TQ=%d\n",
           r->prompt_tokens, r->chunk, p->nchunks, p->cfg.prefill_cards, p->cfg.attn_tq);
    printf("PREFILL %.1f tok/s      (%.2f ms for the prompt)\n",
           r->prefill_toks, r->prefill_ms);
    printf("TTFT    %.2f ms\n", r->ttft_ms);
    if (!p->cfg.verbose) { printf("(PF_VERBOSE=1 for the stage decomposition)\n"); return; }
    double tot = 0.0;
    for (int s = 0; s < PF_S_COUNT; s++)
        for (int d = 0; d < P92_P2P_NGPU; d++) tot += p->timers[d].ms[s];
    printf("\n  %-9s %10s %8s %11s\n", "stage", "ms total", "percent", "ms/token");
    for (int s = 0; s < PF_S_COUNT; s++) {
        double v = 0.0;
        for (int d = 0; d < P92_P2P_NGPU; d++) v += p->timers[d].ms[s];
        if (v <= 0.0) continue;
        printf("  %-9s %10.2f %7.1f%% %11.4f\n", pf_stage_name[s], v,
               100.0 * v / (tot > 0 ? tot : 1), v / r->prompt_tokens);
    }
    printf("  %-9s %10.2f\n", "SUM", tot);
    // Card busy time against wall time says whether the pipeline is overlapping
    // or merely correct. Four cards each busy 1/4 of the wall is a pipeline that
    // never overlapped.
    printf("\n  wall %.2f ms, summed card time %.2f ms, overlap factor %.2fx of %d\n",
           r->prefill_ms, tot, tot / (r->prefill_ms > 0 ? r->prefill_ms : 1), P92_P2P_NGPU);
}

