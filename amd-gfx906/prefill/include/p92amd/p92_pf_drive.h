// Producer-push chunk driver for openPangu prefill.
//
// PRODUCER-PUSH IS THE ARCHITECTURE, not a transport swapped in later. Card n
// computes its layers for a chunk and WRITES the result into card n+1's landing
// buffer; n+1 then reads its own memory. Measured on these MI50s: peer write
// 23.1 GB/s, peer read 3.85 GB/s, local 653.6 GB/s. A consumer that dereferences
// a peer pointer runs 2.6x slower than the host staging it would replace, so
// the pull direction is not a fallback, it is a defect.
//
// THE OVERLAP IS THE POINT, not the handoff latency. With C chunks over 4 layer
// stages, utilisation is C/(C+3): card n starts chunk k+1 the moment it has
// pushed chunk k, so card n+1 consumes chunk k while n is still producing k+1.
// A driver that pushes and then waits has the transport right and the pipeline
// wrong.
//
// STAGE TIMING FROM THE FIRST RUN. The first composed T=512 number must arrive
// already decomposed, or the next decision gets made by staring at VGPR counts.
#ifndef P92AMD_PF_DRIVE_H
#define P92AMD_PF_DRIVE_H

#include "p92amd/p92_pf.h"
#include "p92amd/p92_p2p.h"

enum PfStage {
    PF_S_MHC = 0,     // pre-mix, Sinkhorn, collapse, re-expand
    PF_S_PROJ,        // q_a, q_b, kv_a, o_proj
    PF_S_MOME,        // three width-3 causal convs
    PF_S_ROPE,
    PF_S_ATTN,        // chunked MLA
    PF_S_ROUTE,       // router + device grouping
    PF_S_GATEUP,
    PF_S_REQUANT,
    PF_S_DOWN,
    PF_S_NORM,
    PF_S_HANDOFF,     // producer-push across the card boundary
    PF_S_DSA,         // indexer key cache (every position, always)
    PF_S_DSA_SCORE,   // prefix scoring: the O(n^2) term
    PF_S_DSA_SEL,     // top-2048 radix
    PF_S_DSA_UNI,     // union + mask construction
    PF_S_COUNT
};
static const char *pf_stage_name[PF_S_COUNT] = {
    "mhc", "proj", "mome", "rope", "attn", "route",
    "gateup", "requant", "down", "norm", "handoff", "dsa", "dsascore", "dsasel", "dsauni"
};

// WALL SENSITIVITY. A stage's profiled milliseconds are NOT its recoverable
// milliseconds. In a pipeline with barriers and cross-card handoffs, work that
// sits in slack can carry a large profile bucket and contribute nothing to token
// wall - so attacking the biggest bucket is a coin flip until its sensitivity is
// known.
//
//     recoverable critical-path ms  =  profiled ms  x  wall sensitivity
//
// Measured by DUPLICATING an idempotent stage, never by skipping it. Skipping
// changes the computation and produces an invalid ceiling - this project has
// already burned a probe that way, where a variant doing strictly more work ran
// faster than its own subset because removing a component changed register
// allocation and scheduling. Duplication preserves every result and every token,
// and adds a known quantity of the exact work in question.
//
//     duplicate stage S, observe delta wall / delta stage time
//       ~1.0  fully exposed, on the critical path, worth attacking
//       ~0.0  hidden in slack, the profile bucket is a mirage
//       0<k<1 partially exposed; k is the coefficient to multiply the bucket by
//
// A POSITIVE CONTROL IS MANDATORY. Duplicate a stage known to be exposed - the
// last stage before the final barrier - and require the wall to respond nearly
// proportionally. If it does not, the instrument is broken and every sensitivity
// number it produced is meaningless.
//
// PF_SENS_STAGE selects the stage to duplicate, PF_SENS_REPS how many extra
// times. Both default off, so a sensitivity run and a normal run are the same
// binary.

// Per-card, per-stage accumulation. Events are created ON the card they time -
// recording an event created on another device is undefined and silently
// produced garbage in an earlier probe this session.
// ONE EVENT PAIR A LAUNCH SITE, not one a stage. The first version kept a
// single beg/end per stage and re-recorded them on every layer, so
// pf_stage_accumulate measured the LAST layer's stage and nothing else: with
// ~11 layers a card the reported total was low by roughly that factor, and the
// summed card time came out at 316 ms against a 2502 ms wall. That read as
// "the cards are idle 87% of the time" when the cards were in fact busy. A
// profile bucket from an overwritten event is not a small error, it is a
// different conclusion - so slots are recorded per site and reduced after the
// sync that pf_layers already performs.
#define PF_TM_SLOTS 512
struct PfTimers {
    hipEvent_t    eb[PF_TM_SLOTS], ee[PF_TM_SLOTS];
    unsigned char est[PF_TM_SLOTS];
    int           nslot, overflow;
    double        ms[PF_S_COUNT];
    int        device;
    int        enabled;
};

int  pf_timers_init(PfTimers *t, int device, int enabled);
void pf_timers_free(PfTimers *t);
// Wrap a stage. When disabled these are no-ops, so the timed and untimed builds
// are the same binary and a sweep never compares across builds.
static inline void pf_stage_begin(PfTimers *t, PfStage s, hipStream_t st) {
    if (!t->enabled) return;
    if (t->nslot >= PF_TM_SLOTS) { t->overflow++; return; }
    t->est[t->nslot] = (unsigned char)s;
    hipEventRecord(t->eb[t->nslot], st);
}
static inline void pf_stage_end(PfTimers *t, PfStage s, hipStream_t st) {
    if (!t->enabled) return;
    if (t->nslot >= PF_TM_SLOTS) return;
    hipEventRecord(t->ee[t->nslot], st);
    t->nslot++;
}
void pf_stage_accumulate(PfTimers *t);          // after a sync, adds to ms[]

// The chunk pipeline. Each card runs one of these on its own thread: layers it
// owns, for every chunk, pushing downstream as soon as a chunk clears its last
// owned layer rather than after all chunks are done.
struct PfPipeline {
    P92P2P   p2p;
    PfConfig cfg;
    PfTimers timers[P92_P2P_NGPU];
    int      layer_begin[P92_P2P_NGPU], layer_end[P92_P2P_NGPU];
    int      nchunks;
    int      prompt_tokens;
    int      sens_stage;    // -1 = off, else the PfStage to duplicate
    int      sens_reps;     // extra executions of that stage
};

int  pf_pipeline_init(PfPipeline *p, const PfConfig *cfg, int prompt_tokens);
void pf_pipeline_free(PfPipeline *p);

// Runs the whole prompt and fills r. The ONLY acceptance number is
// r->prefill_toks; everything in stage_ms is diagnosis.
// The HARNESS owns the chunk loop, not the driver: the driver supplies timers,
// the peer transport and the report. pf_sensitivity assumed the opposite and is
// removed - the probe belongs where the loop lives, and it is not a prerequisite
// for first light.

// Prints the decomposition the first composed run must produce:
//   stage   ms/chunk   percent   ms/token
void pf_report(const PfPipeline *p, const PfResult *r);


#endif
