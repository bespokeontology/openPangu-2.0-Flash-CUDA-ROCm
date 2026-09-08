# openPangu Flash92 native CUDA engine
#
# Third-party include paths must be supplied at configure time, e.g.
#   cmake -S . -B build -DP92_CUTLASS_INCLUDE=/path/to/cutlass/include \
#     -DP92_CUTLASS_TOOLS_INCLUDE=/path/to/cutlass/tools/util/include \
#     -DP92_FLASHINFER_INCLUDE=/path/to/flashinfer/include
# Target architecture is SM 12.1a (NVIDIA GB10 / DGX Spark).

cmake_minimum_required(VERSION 3.24)

project(openpangu_flash92_native LANGUAGES CXX CUDA)

set(P92_CUTLASS_INCLUDE "" CACHE PATH "CUTLASS include directory")
set(P92_CUTLASS_TOOLS_INCLUDE "" CACHE PATH "CUTLASS tools/util include directory")
set(P92_FLASHINFER_INCLUDE "" CACHE PATH "FlashInfer include directory")
set(P92_AGENT_RUNTIME_DIR "" CACHE PATH "Optional agent runtime directory")

find_package(CUDAToolkit REQUIRED)
find_library(PCRE2_8_LIBRARY pcre2-8 REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(p92_catalog STATIC src/catalog.cpp)
target_include_directories(p92_catalog PUBLIC include)
target_compile_options(p92_catalog PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_library(p92_auxiliary STATIC src/auxiliary.cpp)
target_include_directories(p92_auxiliary PUBLIC include)
target_link_libraries(p92_auxiliary PUBLIC p92_catalog CUDA::cudart)
target_compile_options(p92_auxiliary PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_library(p92_tokenizer STATIC src/tokenizer.cpp)
target_include_directories(p92_tokenizer PUBLIC include)
target_compile_definitions(p92_tokenizer PRIVATE PCRE2_CODE_UNIT_WIDTH=8)
target_link_libraries(p92_tokenizer PUBLIC ${PCRE2_8_LIBRARY})
target_compile_options(p92_tokenizer PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_library(p92_nvfp4_artifact STATIC src/nvfp4_artifact.cpp)
target_include_directories(p92_nvfp4_artifact PUBLIC include)
target_compile_options(p92_nvfp4_artifact PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_library(p92_resident STATIC src/resident.cpp)
target_include_directories(p92_resident PUBLIC include)
target_link_libraries(p92_resident PUBLIC p92_nvfp4_artifact p92_kernels CUDA::cudart)
target_compile_options(p92_resident PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(p92_nvfp4_artifact_inspect src/nvfp4_artifact_main.cpp)
target_link_libraries(p92_nvfp4_artifact_inspect PRIVATE p92_nvfp4_artifact)
target_compile_options(p92_nvfp4_artifact_inspect PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(p92_nvfp4_swizzle src/swizzle_main.cpp)
target_include_directories(p92_nvfp4_swizzle PRIVATE include)
target_link_libraries(p92_nvfp4_swizzle PRIVATE p92_nvfp4_artifact p92_kernels CUDA::cudart)
target_compile_options(p92_nvfp4_swizzle PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(p92_nvfp4_swizzle PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_executable(p92_catalog_inspect src/catalog_main.cpp)
target_link_libraries(p92_catalog_inspect PRIVATE p92_catalog)
target_compile_options(p92_catalog_inspect PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(p92_tokenizer_test tests/tokenizer_test.cpp)
target_link_libraries(p92_tokenizer_test PRIVATE p92_tokenizer)
target_compile_options(p92_tokenizer_test PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(p92_nvfp4_inventory src/nvfp4_inventory.cpp)
target_link_libraries(p92_nvfp4_inventory PRIVATE p92_catalog)
target_compile_options(p92_nvfp4_inventory PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_library(p92_kernels STATIC
    cuda/dsa_ops.cu
    cuda/nvfp4_grouped.cu
    cuda/nvfp4_ops.cu
    cuda/model_ops.cu
    cuda/nvfp4_tensorcore.cu
)
target_include_directories(p92_kernels PUBLIC include)
target_include_directories(p92_kernels SYSTEM PRIVATE
    ${P92_CUTLASS_INCLUDE}
    ${P92_CUTLASS_TOOLS_INCLUDE}
    ${P92_FLASHINFER_INCLUDE}
)
target_link_libraries(p92_kernels PUBLIC CUDA::cudart CUDA::cublas)
target_compile_definitions(p92_kernels PRIVATE
    ENABLE_BF16
    ENABLE_FP4
    CUTLASS_ENABLE_GDC_FOR_SM100=1
    FLASHINFER_ENABLE_FP4_E2M1
    FLASHINFER_ENABLE_FP8_E8M0
)
target_compile_options(p92_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-gencode=arch=compute_121a,code=sm_121a;--expt-relaxed-constexpr;--use_fast_math;-Xcompiler=-Wall,-Wextra,-Werror>
)
set_target_properties(p92_kernels PROPERTIES
    CUDA_ARCHITECTURES OFF
    POSITION_INDEPENDENT_CODE ON
)

add_library(p92_executor STATIC src/executor.cpp)
target_include_directories(p92_executor PUBLIC include)
target_link_libraries(p92_executor PUBLIC p92_resident p92_kernels CUDA::cudart)
target_compile_options(p92_executor PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_library(p92_model STATIC src/model.cpp)
target_include_directories(p92_model PUBLIC include)
target_link_libraries(p92_model PUBLIC
    p92_auxiliary p92_executor p92_resident p92_kernels CUDA::cudart
)
target_compile_options(p92_model PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(p92_model PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_executable(p92_nvfp4_projection_test cuda/nvfp4_projection_test.cu)
target_include_directories(p92_nvfp4_projection_test PRIVATE include)
target_link_libraries(p92_nvfp4_projection_test PRIVATE
    p92_catalog p92_nvfp4_artifact p92_kernels CUDA::cudart
)
target_compile_options(p92_nvfp4_projection_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-gencode=arch=compute_121a,code=sm_121a;--use_fast_math;-Xcompiler=-Wall,-Wextra,-Werror>
)
set_target_properties(p92_nvfp4_projection_test PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA
)

add_executable(p92_nvfp4_pack cuda/nvfp4_pack.cu)
target_include_directories(p92_nvfp4_pack PRIVATE include)
target_link_libraries(p92_nvfp4_pack PRIVATE p92_catalog CUDA::cudart)
target_compile_options(p92_nvfp4_pack PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-gencode=arch=compute_121a,code=sm_121a;--use_fast_math;-Xcompiler=-Wall,-Wextra,-Werror>
)
set_target_properties(p92_nvfp4_pack PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA
)

add_executable(p92_mhc_state_test tests/mhc_state_test.cu)
target_include_directories(p92_mhc_state_test PRIVATE include)
target_link_libraries(p92_mhc_state_test PRIVATE p92_catalog p92_kernels CUDA::cudart)
target_compile_options(p92_mhc_state_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-gencode=arch=compute_121a,code=sm_121a;--use_fast_math;-Xcompiler=-Wall,-Wextra,-Werror>
)
set_target_properties(p92_mhc_state_test PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA
)

add_executable(p92_resident_test tests/resident_test.cpp)
target_include_directories(p92_resident_test PRIVATE include)
target_link_libraries(p92_resident_test PRIVATE p92_resident CUDA::cudart)
target_compile_options(p92_resident_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(p92_resident_test PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_executable(p92_projection_rows_test tests/projection_rows_test.cpp)
target_include_directories(p92_projection_rows_test PRIVATE include)
target_link_libraries(p92_projection_rows_test PRIVATE p92_executor CUDA::cudart)
target_compile_options(p92_projection_rows_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(p92_projection_rows_test PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_executable(p92_nvfp4_tactics_test tests/nvfp4_tactics_test.cpp)
target_include_directories(p92_nvfp4_tactics_test PRIVATE include)
target_link_libraries(p92_nvfp4_tactics_test PRIVATE p92_resident p92_kernels CUDA::cudart)
target_compile_options(p92_nvfp4_tactics_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(p92_nvfp4_tactics_test PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_executable(p92_mla_decode_test tests/mla_decode_test.cu)
target_include_directories(p92_mla_decode_test PRIVATE include)
target_link_libraries(p92_mla_decode_test PRIVATE p92_auxiliary p92_kernels CUDA::cudart)
target_compile_options(p92_mla_decode_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-gencode=arch=compute_121a,code=sm_121a;--use_fast_math;-Xcompiler=-Wall,-Wextra,-Werror>
)
set_target_properties(p92_mla_decode_test PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA
)

add_executable(p92_mla_rows_test tests/mla_rows_test.cu)
target_include_directories(p92_mla_rows_test PRIVATE include)
target_link_libraries(p92_mla_rows_test PRIVATE p92_auxiliary p92_kernels CUDA::cudart)
target_compile_options(p92_mla_rows_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-gencode=arch=compute_121a,code=sm_121a;--use_fast_math;-Xcompiler=-Wall,-Wextra,-Werror>
)
set_target_properties(p92_mla_rows_test PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA
)

add_executable(p92_dsa_test tests/dsa_test.cu)
target_include_directories(p92_dsa_test PRIVATE include)
target_link_libraries(p92_dsa_test PRIVATE p92_kernels CUDA::cudart)
target_compile_options(p92_dsa_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-gencode=arch=compute_121a,code=sm_121a;--use_fast_math;-Xcompiler=-Wall,-Wextra,-Werror>
)
set_target_properties(p92_dsa_test PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA
)

add_executable(p92_mome_rows_test tests/mome_rows_test.cu)
target_include_directories(p92_mome_rows_test PRIVATE include)
target_link_libraries(p92_mome_rows_test PRIVATE p92_kernels CUDA::cudart)
target_compile_options(p92_mome_rows_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-gencode=arch=compute_121a,code=sm_121a;--use_fast_math;-Xcompiler=-Wall,-Wextra,-Werror>
)
set_target_properties(p92_mome_rows_test PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    LINKER_LANGUAGE CUDA
)

add_executable(p92_layer0_forward_test tests/layer0_forward_test.cpp)
target_include_directories(p92_layer0_forward_test PRIVATE include)
target_link_libraries(p92_layer0_forward_test PRIVATE
    p92_auxiliary p92_executor p92_resident p92_kernels CUDA::cudart
)
target_compile_options(p92_layer0_forward_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(p92_layer0_forward_test PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_executable(p92_moe_forward_test tests/moe_forward_test.cpp)
target_include_directories(p92_moe_forward_test PRIVATE include)
target_link_libraries(p92_moe_forward_test PRIVATE
    p92_auxiliary p92_executor p92_resident p92_kernels CUDA::cudart
)
target_compile_options(p92_moe_forward_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(p92_moe_forward_test PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_executable(p92_full_forward_test tests/full_forward_test.cpp)
target_link_libraries(p92_full_forward_test PRIVATE p92_model)
target_compile_options(p92_full_forward_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(p92_full_forward_test PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_executable(p92_prefill_test tests/prefill_test.cpp)
target_include_directories(p92_prefill_test PRIVATE include)
target_link_libraries(p92_prefill_test PRIVATE p92_model CUDA::cudart)
target_compile_options(p92_prefill_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(p92_prefill_test PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_executable(p92_decode_profile tests/decode_profile.cpp)
target_include_directories(p92_decode_profile PRIVATE include)
target_link_libraries(p92_decode_profile PRIVATE p92_model CUDA::cudart)
target_compile_options(p92_decode_profile PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(p92_decode_profile PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_executable(p92_mtp_test tests/mtp_test.cpp)
target_include_directories(p92_mtp_test PRIVATE include)
target_link_libraries(p92_mtp_test PRIVATE p92_model p92_tokenizer CUDA::cudart)
target_compile_options(p92_mtp_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(p92_mtp_test PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_executable(p92_chat src/chat_main.cpp)
target_link_libraries(p92_chat PRIVATE p92_model p92_tokenizer)
target_compile_options(p92_chat PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(p92_chat PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_executable(p92_chat_mtp src/chat_main.cpp)
target_compile_definitions(p92_chat_mtp PRIVATE P92_CHAT_MTP=1)
target_link_libraries(p92_chat_mtp PRIVATE p92_model p92_tokenizer)
target_compile_options(p92_chat_mtp PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(p92_chat_mtp PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_executable(p92_serve_mtp src/serve_main.cpp)
target_link_libraries(p92_serve_mtp PRIVATE p92_model p92_tokenizer)
target_compile_options(p92_serve_mtp PRIVATE -Wall -Wextra -Wpedantic -Werror)
set_target_properties(p92_serve_mtp PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# The tool/controller host is model-independent and never allocates CUDA.
set(NATIVE_AGENT_RUNTIME_ROOT
    "${P92_AGENT_RUNTIME_DIR}"
    CACHE PATH "Shared native agent runtime")
add_subdirectory(${NATIVE_AGENT_RUNTIME_ROOT}
                 ${CMAKE_CURRENT_BINARY_DIR}/native-agent-runtime
                 EXCLUDE_FROM_ALL)

add_executable(p92_agent src/agent_main.cpp)
target_link_libraries(p92_agent PRIVATE
    p92_model p92_tokenizer native_agent_core)
target_compile_options(p92_agent PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
set_target_properties(p92_agent PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_executable(p92_state_snapshot_test tests/state_snapshot_test.cpp)
target_link_libraries(p92_state_snapshot_test PRIVATE
    p92_model p92_tokenizer)
target_compile_options(p92_state_snapshot_test PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
set_target_properties(p92_state_snapshot_test PROPERTIES
    LINKER_LANGUAGE CUDA
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
