# 21 — Ingest: the actual blocker for 262K, and a defect in the long-context runs

## 1. Decode residency is not the 262K blocker (measured)

`tests/p92_generate.hip` sizes the decode caches directly from `MAXPOS`:

```
w.cap = w.dsa ? MAXPOS : 512;                  // SWA layers stay at 512
cache_kv   = cap * KVLORA * 2
cache_rope = cap * ROPE * 2
ix_cache   = P92_IDX_DIM * cap * 2             // bf16 index keys
c.scores   = MAXPOS * 4 ;  c.sel = max(MAXPOS,2048) * 4
```

At `MAXPOS=262144` that is ~369 MB per DSA layer, ~1.5 GB per card for its 4 DSA
layers. Measured, twice:

| run | MAXPOS | result |
|---|---|---|
| 4 tokens | 262144 | loads all 46 layers, **69.14 tok/s** |
| 8 tokens | 262144 | **70.23 tok/s**, 14.24 ms/token steady |

So the earlier statement that 128K/256K "exceed VRAM on the DSA cache allocation"
was wrong about the decoder. Allocation and residency at 262,144 are fine.

## 2. The blocker is ingest, and its shape is now precise

`tests/p92_chat.hip` has **no prefill**: the prompt is walked one position per
generation step, teacher-forced:

```
token = ((size_t)(step+1) < prompt.size()) ? prompt[step+1] : next;   // :749
```

argv[4] is the **total number of positions**, not "tokens to generate". Generation
only starts once the whole prompt has been walked. Ingest therefore costs a full
decode step per prompt token: ~14.2 ms at short context, ~20 ms past 5.7K, so
262,144 positions is roughly **1.5 hours before the first generated token**.
Ingesting 131,072 positions with `argv[4]=8` is what a first attempt at this
measured: it processed 8 positions and looked like a 17-second 131K ingest. The
rule to remember is `argv[4] = prompt_tokens + tokens_to_generate`.

## 3. The fast chunked prefill exists but is not wired to the decoder

`prefill/tests/p92_pf_bench.hip` has a chunk pipeline (`cfg.chunk`, `attn_tq`,
`prefill_cards`) and writes the **same cache layout at an advancing base**:
`prefill/src/p92_pf_mla.hip` stores `cache_kv + (base+t)*PM_LATENT` and
`cache_rope + (base+t)*PM_ROPE`, with the bench allocating `cap*KVLORA*2`,
`cap*ROPE*2`, `P92_IDX_DIM*cap*2` exactly as the decoder does. The 698 tok/s /
65,536-token figure was produced by this path.

What it does **not** have: a real token source — its prompt is synthetic,
`tokens[i] = (START+i) % 151552` (`p92_pf_bench.hip:523`) — no persistence, and no
handoff into the decode process. Those three things are the build:

1. feed the chunk pipeline a real token file and advance `base` across chunks to
   `MAXPOS`;
2. hand `cache_kv`/`cache_rope`/`ix_cache` to the decoder, in-process (ingest then
   decode in one binary) or by dump/load — the dump/load form is also the first
   real instance of the cold-state tier;
3. then retrieval probes at 25/50/75/95% depth.

## 4. Defect: the long-context prompt files are out of vocabulary

`/tmp/prompt62k.i64` and the 131,072-token file built from it contain token ids
from **1 to 154,841**. Pangu's embedding table has **151,552 rows** (head
`151552x2560`) and the tokenizer asserts 148,899 entries, so every id above
151,551 indexes **past the end of the embedding matrix**. The harness does no range
check; `embed_ptr(token)` simply computes an address past the tensor.

Consequences, stated plainly:

- The **timing** of the long-context decode curve is still meaningful (48.9 tok/s at
  43,135 positions is a real measurement at that length — the kernels do the same
  work whatever the token id).
- The **content** of those runs is meaningless, and no retrieval claim can rest on
  them. The 62,144-token "supported context" figure is likewise a capacity result
  measured on invalid ids, not a semantic result.
- The ids look like another tokenizer's (max 154,841 is above Pangu's padded head),
  i.e. the file was built with the wrong tokenizer rather than being a layout error.

Fix, next: produce long prompt files by tokenizing real text with the model's own
tokenizer (`tools/p92_tokenizer.cpp`, which already implements it, including the
pangu regex), write int64 ids, and only then make semantic claims.

## In flight

A 70,008-position run (`MAXPOS=262144`, 70,000 prompt positions + 8 generated) is
ingesting now. It crosses 2^16 = 65,536 for the first time, which is the one
boundary that would break a 16-bit position or slot index anywhere in the caches;
its tail will report the decode cost at ~70K context.
