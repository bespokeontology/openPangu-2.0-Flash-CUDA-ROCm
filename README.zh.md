# openPangu Flash92 — C++/CUDA 与 C++/HIP ROCm 推理引擎

[English](README.md) | **中文** | [Русский](README.ru.md)

面向华为 `openPangu-2.0-Flash`（总参数 920 亿，每 token 约 60 亿激活）的原生 C++/HIP
推理引擎，目标硬件为 4× AMD MI50 / Radeon Pro VII（gfx906）。运行时不包含 Python、
不包含任何框架、热路径不调用 rocBLAS、不含通用 GEMM 回退路径。

这是同一项目的第二个原生后端。仓库根目录的 DGX Spark CUDA 后端是较早的实现，
原样保留，作为历史基线与对照。

## 状态

- **解码：已发布权威结果，纯原生目标解码，MTP 关闭。**
  gfx906、四卡、12/12/11/11 层归属，稳态 69.92 tok/s。
- **Prefill：已发布权威结果（SPG2）：4K / 8K / 16K / 32K / 64K 提示长度分别为
  710 / 731 / 732 / 725 / 698 tok/s。**
- **MTP：实验性工作，不在本次发布性能口径内。** 开发分支上存在原生三头草稿路径与
  投机状态机；固定 T 的批量目标验证器尚未完成。见 `amd-gfx906/docs/MTP_STATUS.md`。

## 后端对照

| 后端 | 硬件 | 解码模式 | MTP | 解码 tok/s | Prefill tok/s | 状态 |
|---|---|---|---:|---:|---:|---|
| NVIDIA CUDA | DGX Spark GB10 | MTP / 投机解码 | 开 | 52.10（上下文 123，生成 96，接受率 82.7%） | 51.4 @ 3,807 tokens | 历史基线，保留 |
| NVIDIA CUDA | DGX Spark GB10 | 纯目标解码 | 关 | 21.01（上下文 331） | 51.4 @ 3,807 tokens | 历史基线，保留 |
| AMD gfx906 | 4× MI50 / Pro VII | 纯目标解码 | 关 | 69.92（上下文 512，固定 token 流） | 710 / 731 / 732 / 725 / 698（4K–64K） | 当前 |

上表 AMD 解码结果不使用 MTP 或投机解码；上表 DGX Spark 解码结果使用。

完整表格、方法与出处见 `amd-gfx906/docs/BENCHMARKS.md`。

## 硬件与软件

| 项目 | 值 |
|---|---|
| GPU | 4× AMD MI50 / Radeon Pro VII，gfx906，60 CU，wave64 |
| 显存 | 每卡 17,163,091,968 B（可用 15.98 GiB），合计 63.94 GiB |
| 频率 | 1700 MHz（基准锁强制 DPM `profile_peak`） |
| 主机 | Ubuntu，Linux 6.8 |
| ROCm | /opt/rocm（HIP，clang 17） |
| 构建 | `hipcc --offload-arch=gfx906 -O3 -std=c++17` |

## 目录

```
amd-gfx906/
  decode/     主干解码引擎（p92-amd 树）：src/、include/、tests/、tools/、receipts/
  prefill/    Prefill 引擎（p92-prefill 树）：src/、include/、tests/、receipts
  docs/       AMD_IMPLEMENTATION.md、BENCHMARKS.md、AMD_KERNEL_NOTES.md、
              PREFILL_REPORT.md、MTP_STATUS.md
```

## 权重

两个引擎都读取 NVFP4 容器 `P92FP41`（`manifest.bin`、`weights.nvfp4`、
`scales.e4m3`），以及来自官方检查点的 BF16 张量（norm、mHC 参数、压缩 KV 投影、
逐通道卷积、注意力锚点、路由及其校正偏置、token 嵌入）。容器格式由消费者定义：
`decode/tests/p92_generate.hip`（`MHdr`/`MRec`、`nv_up`）、
`decode/src/p92_nvfp4_dot4.hip`（E2M1 半字节顺序、每 16 元素一组 UE4M3 缩放）、
`decode/tools/p92_pack_arena.cpp`（arena 布局）。

生成 MTP 容器的量化器已包含在 `decode/tools/mtp_quant.cpp`，其中给出确切的打包流程；
主干容器由同一流程按 `docs/AMD_IMPLEMENTATION.md` 中记录的张量普查生成。

## 构建（解码）

```
cd amd-gfx906/decode
/opt/rocm/bin/hipcc --offload-arch=gfx906 -O3 -std=c++17 -Iinclude \
  tests/p92_generate.hip -o p92_gen -lpthread
```

运行：`./p92_gen <checkpoint> <artifact> <arena> <tokens> <maxpos> <start-token>`

## 构建（Prefill）

```
cd amd-gfx906/prefill
/opt/rocm/bin/hipcc --offload-arch=gfx906 -O3 -std=c++17 -Iinclude \
  -I../decode/include tests/p92_pf_bench.hip src/p92_pf_shuffle.hip \
  src/p92_p2p.hip src/p92_pf_drive.hip -o p92_pf_bench -lpthread
```

在机器锁下运行：`~/q27bench ./p92_pf_bench <checkpoint> <artifact> <arena> <prompt> <maxpos>`

## 模型与许可

仅含源码，不含任何模型权重。上游华为署名与 OpenPangu Model License Agreement
Version 2.0 见仓库 `NOTICE` 与 `legal/`。
