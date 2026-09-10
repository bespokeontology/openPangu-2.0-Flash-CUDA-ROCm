# 23 — Retrieval at 74,891 positions (partial), and the persistent server

## 1. Retrieval recall at depth is not perfect, and that is the finding

Same 242,934 B corpus, needles at 25/50/75/95% depth, one question per run:

| ingest positions | question | needle at 50% (basalt) | rate |
|---|---|---|---|
| 32,852 | "basalt vault?" | **found and quoted verbatim** | 50.11 tok/s |
| 74,891 | "basalt vault?" | **not found** — model scanned, quoted amber and cobalt instead | 49.14 tok/s |

At 74,891 positions the model answered:

> user is asking for the access code to the basalt vault. I need to look through the
> provided documents to find any mention of a basalt vault and its access code.
>
> Scanning the documents... I see references to "amber vault" and "cobalt vault" but
> not "basalt vault" specifically. Let me check the numbers mentioned in the
> documents.
>
> In the first document, there's a line: "The amber vault access code is 41-2739."
> In the second document, there's a line: "The cobalt vault access code is 63-8157."

So the context at depth is genuinely used (25% and 75% needles quoted verbatim at
74,891 positions), but the specific needle at 50% was missed. This is **not** the
same as a broken engine: attention is reading the right region — it is a recall
limit that has to be characterised rather than assumed away.

Caveat on sample size: one question per depth, one run each. The 32,852/50% cell and
the 74,891/50% cell are single observations and the model's scan strategy varies
between runs. The honest statement is: *retrieval at depth works in this engine and
degrades at 75K; the shape of the degradation is not yet measured.*

## 2. The measurement that fixes this: ask several questions per ingest

Asking one question per run costs a full re-ingest (11 min at 32K, 26 min at 74K,
~93 min at 262K). `tests/p92_serve.hip` (new; built as `p92_serve`, 393,136 B,
sha256 `78c0b4c0438ef82e9a34c6f2ea73b835db4c78f50d5dabebbd668d5725efe4`) removes that:

- `P92_SERVE=1` keeps the ingested context resident and reads further requests from
  stdin, appending each to the context and answering it in place;
- a `force` deque holds tokens to feed verbatim and drains into generation, which is
  what makes the prompt walk and the open-ended generation the same loop;
- positions advance monotonically across requests (`base_pos`), so request *n* attends
  to everything ingested before it;
- `P92_MAXNEW` bounds each answer.

This turns a 93-minute ingest into a one-time cost that answers many questions, which
is the whole usability argument for 262K context: you pay the ingest once and then
ask.

### Verified at runtime (smoke test, rc=0)

A 130-token corpus with the amber and basalt needles, two questions piped in on
stdin, 64 new tokens each. The server's own markers show positions advancing across
requests — that is the context being reused, not rebuilt:

```
[serve] request 30 tokens, at position 194, budget 94
[serve] request 30 tokens, at position 288, budget 94
```

Answer 1: `...the context given in the initial message: "The amber vault access code is 41-2739."`

Answer 2, asked after request 1 had already been appended to the live context, with no
re-ingest of any kind:
```
First, the user is asking for the access code for the basalt vault. I need to recall
the information from the initial context provided in the conversation.
The initial context says: "The basalt vault access code is 58-1904."
```

Both needles retrieved, in the second and third requests of one process, at 288
positions of accumulated context.

## 3. Lock discipline held

The machine lock refused both the 262K probe and the server test at 16:46:52 because
the Qwen lane held the box (`q27bench` exit 3, holder pid running `q27_gen`). Nothing
was run unlocked and nothing was taken from that lane. Every GPU command in the
current chain is wrapped in a lock-aware retry, so an overlapping window delays the
run instead of silently killing it.

## 4. Where the 262K proof stands

- allocation and residency at `MAXPOS=262144`: **verified** (receipts 21, 22);
- positions past 2^16: **verified** to 70,008 (receipt 22);
- retrieval at depth: **verified** at 32,852, **partial** at 74,891 (this receipt);
- ingest of a real 261,071-token corpus then four questions against it: **queued**.

The gap that remains is ingest throughput: ~20.5 ms/position is decode speed, and the
chunked 698 tok/s prefill cannot be used as an ingest because it launches no indexer
above 2,048 positions (receipt 22 §3). Until the indexer runs inside the chunk, large
contexts are affordable once, not repeatedly.
