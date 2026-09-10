# openPangu Flash92 — движки C++/CUDA и C++/HIP ROCm

[English](README.md) | [中文](README.zh.md) | **Русский**

Нативный движок вывода на C++/HIP для модели Huawei `openPangu-2.0-Flash`
(92 млрд параметров всего, ~6 млрд активных на токен) на 4× AMD MI50 / Radeon Pro VII
(gfx906). Без Python, без фреймворков, без rocBLAS на горячем пути, без универсального
GEMM-фолбэка.

Это второй нативный бэкенд одного и того же проекта. Бэкенд DGX Spark CUDA в корне
репозитория — более ранняя реализация; он сохранён без изменений как историческая
база для сравнения.

## Состояние

- **Декодирование: опубликованный результат, чистое нативное декодирование цели,
  MTP выключен.** gfx906, четыре карты, распределение слоёв 12/12/11/11,
  установившаяся скорость 69.92 ток/с.
- **Prefill: опубликованный результат (SPG2): 710 / 731 / 732 / 725 / 698 ток/с при
  длине подсказки 4K / 8K / 16K / 32K / 64K.**
- **MTP: экспериментальная работа, не входит в опубликованное утверждение о
  производительности.** В ветке разработки существуют нативный трёхголовый черновик и
  машина состояний спекулятивного декодирования; пакетный верификатор цели с
  фиксированным T не завершён. См. `amd-gfx906/docs/MTP_STATUS.md`.

## Сравнение бэкендов

| Бэкенд | Оборудование | Режим декодирования | MTP | Декод. ток/с | Prefill ток/с | Статус |
|---|---|---|---:|---:|---:|---|
| NVIDIA CUDA | DGX Spark GB10 | MTP / спекулятивное | вкл | 52.10 (контекст 123, 96 сген., принятие 82.7%) | 51.4 @ 3 807 токенов | историческая база |
| NVIDIA CUDA | DGX Spark GB10 | чистое декодирование цели | выкл | 21.01 (контекст 331) | 51.4 @ 3 807 токенов | историческая база |
| AMD gfx906 | 4× MI50 / Pro VII | чистое декодирование цели | выкл | 69.92 (контекст 512, фиксированный поток) | 710 / 731 / 732 / 725 / 698 (4K–64K) | текущий |

Результат AMD для декодирования, приведённый выше, не использует MTP или
спекулятивное декодирование; результат DGX Spark, приведённый выше, использует.

Полные таблицы, методика и источники: `amd-gfx906/docs/BENCHMARKS.md`.

## Оборудование и ПО

| Параметр | Значение |
|---|---|
| GPU | 4× AMD MI50 / Radeon Pro VII, gfx906, 60 CU, wave64 |
| VRAM | 17 163 091 968 Б на карту (15.98 ГиБ доступно), 63.94 ГиБ суммарно |
| частоты | 1700 МГц (блокировка бенчмарка требует DPM `profile_peak`) |
| хост | Ubuntu, Linux 6.8 |
| ROCm | /opt/rocm (HIP, clang 17) |
| сборка | `hipcc --offload-arch=gfx906 -O3 -std=c++17` |

## Структура

```
amd-gfx906/
  decode/     движок декодирования (дерево p92-amd): src/, include/, tests/, tools/, receipts/
  prefill/    движок prefill (дерево p92-prefill): src/, include/, tests/, receipts
  docs/       AMD_IMPLEMENTATION.md, BENCHMARKS.md, AMD_KERNEL_NOTES.md,
              PREFILL_REPORT.md, MTP_STATUS.md
```

## Веса

Оба движка читают контейнер NVFP4 `P92FP41` (`manifest.bin`, `weights.nvfp4`,
`scales.e4m3`) и BF16-тензоры из официального чекпойнта (нормы, параметры mHC,
проекции сжатого KV, depthwise-свёртки, attention-синки, роутеры и их
корректирующие смещения, эмбеддинги токенов). Формат контейнера задаётся его
потребителями: `decode/tests/p92_generate.hip` (`MHdr`/`MRec`, `nv_up`),
`decode/src/p92_nvfp4_dot4.hip` (порядок полубайтов E2M1, масштаб UE4M3 на группу
из 16 элементов), `decode/tools/p92_pack_arena.cpp` (раскладка arena).

Квантователь, создавший контейнер MTP, включён в `decode/tools/mtp_quant.cpp` и
показывает точную процедуру упаковки; контейнер основной модели создаётся той же
процедурой по переписи тензоров, описанной в `docs/AMD_IMPLEMENTATION.md`.

## Сборка (декодирование)

```
cd amd-gfx906/decode
/opt/rocm/bin/hipcc --offload-arch=gfx906 -O3 -std=c++17 -Iinclude \
  tests/p92_generate.hip -o p92_gen -lpthread
```

Запуск: `./p92_gen <checkpoint> <artifact> <arena> <tokens> <maxpos> <start-token>`

## Сборка (prefill)

```
cd amd-gfx906/prefill
/opt/rocm/bin/hipcc --offload-arch=gfx906 -O3 -std=c++17 -Iinclude \
  -I../decode/include tests/p92_pf_bench.hip src/p92_pf_shuffle.hip \
  src/p92_p2p.hip src/p92_pf_drive.hip -o p92_pf_bench -lpthread
```

Запуск под машинной блокировкой:
`~/q27bench ./p92_pf_bench <checkpoint> <artifact> <arena> <prompt> <maxpos>`

## Модель и лицензия

Только исходный код; веса модели не включены. Атрибуция Huawei и OpenPangu Model
License Agreement Version 2.0 — в `NOTICE` и каталоге `legal/`.
