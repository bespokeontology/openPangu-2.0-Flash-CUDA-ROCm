# Assembling the auxiliary checkpoint

The engine loads quantized projection weights from the NVFP4 artifact and a small set
of BF16/F32 tensors from a separate checkpoint directory. The second set is small; the
official 200 GB checkpoint is not needed in full.

## What is required

Derived from the tensor bindings in `src/model.cpp`.

Per trunk layer 0-45:

- `input_layernorm.weight`, `post_attention_layernorm.weight`,
  `pre_mlp_layernorm.weight`, `post_mlp_layernorm.weight`
- `block_post_layernorm.weight` on layers 0, 4, 9, 14, 19, 24, 29, 34, 39 only
- `attn_mhc_module.{phi.weight,norm_gamma,branch_alpha,branch_beta}` and the same four
  under `mlp_mhc_module`
- `self_attn.kv_a_proj_with_mqa.weight`, `self_attn.kv_b_proj.weight`
- `self_attn.q_a_layernorm.weight`, `self_attn.kv_a_layernorm.weight`
- `self_attn.qa_conv.weight`, `self_attn.compresskv_conv.weight`, `self_attn.o_conv.weight`
- `self_attn.param_sink_k_pe`
- `self_attn.indexer.k_norm.weight` and `self_attn.indexer.weights_proj.weight` on
  non-sliding layers (`index % 3 == 0`)
- `mlp.gate.weight` and `mlp.e_score_correction_bias` on MoE layers (2 and above)

Global: `model.embed_tokens.weight`, `model.norm.weight`,
`model.merge_mhc_module.{phi.weight,norm_gamma,branch_alpha_pre,branch_beta_pre}`.

MTP layers 46-48 additionally require `enorm.weight`, `hnorm.weight` and
`shared_head.norm.weight`; they bind no indexer tensors.

Everything else — `q_a_proj`, `q_b_proj`, `o_proj`, `indexer.wq_b`, `indexer.wk`,
`lm_head`, the dense MLP, the routed experts and the shared expert — comes from the
NVFP4 artifact.

## Fetching

`scripts/fetch_auxiliary_tensors.py` reads `model.safetensors.index.json` from the
upstream repository, requests each shard's safetensors header, and issues one HTTP
range request per required tensor. It writes the results into
`model-00001.safetensors` and creates empty well-formed shards for the rest, because
`src/catalog.cpp` iterates a fixed 50-shard range.

Result: 1,101 tensors for the trunk (about 1.6 GB, of which 776 MB is the token
embedding) and 54 tensors for the MTP layers.

## Tokenizer

Fetch `tokenizer.json` from the upstream repository alongside the tensors. The engine
validates it against the official shape and refuses anything else:

| field | required |
|---|---|
| base vocabulary | 148,899 |
| added tokens | 701 |
| merges | 148,643 |
| reverse table | 151,552 |

`merges` must be an array of two-element arrays. The upstream file satisfies all four
conditions as shipped; `p92_tokenizer_test <tokenizer.json>` reports
`P92_TOKENIZER_OK` when it is correct.
