# 19 - TEXT-MODE CHAT (tokenizer + chat template + sampling)

The engine now takes TEXT and answers in TEXT. No Python in the path.

## Command
echo "Explain Kolmogorov complexity in two sentences." | P92_TEMP=0.6 P92_TOPK=50 P92_TOPP=0.95 \
  ~/q27bench ./p92_chat <checkpoint> <artifact> <arena> 160 4096 148899 -

argv[7] = "-" reads the prompt from stdin; a path instead reads int64 token ids.
P92_TEMP=0 selects the original greedy argmax path (the banked 69.92 rung).

## Components
| piece | source |
|---|---|
| BPE tokenizer | tools/p92_tokenizer.cpp + include/p92/tokenizer.h (this model's tokenizer.json; pcre2 pre-tokenizer) |
| chat template | p92::Tokenizer::chat_prompt(user, system, thinking) |
| sampling | temperature / top-k / top-p over the full logits in the decode harness (host side) |
| streaming | decode_token per generated id, flushed |

## Measured (frozen decode path, MTP off)
prompt: "Explain Kolmogorov complexity in two sentences."
answer (verbatim, first sentence): "Sentence 1: Kolmogorov complexity of a string is the length of the shortest computer program that outputs that string."
steady decode 14.82 ms/token = 67.47 tok/s at context 512 (T=0.6, top-k 50, top-p 0.95).

Sampling removes the repetition collapse the greedy path shows on this prompt family;
greedy remains available for reproducing the historical rung.
