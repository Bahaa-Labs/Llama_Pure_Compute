cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(LlamaPureCompute LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES "86")
endif()
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()
find_package(CUDAToolkit REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
	COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.__path__[0])"
	OUTPUT_VARIABLE TORCH_INSTALL_PREFIX
	OUTPUT_STRIP_TRAILING_WHITESPACE
	RESULT_VARIABLE TORCH_PREFIX_RESULT
)
if(NOT TORCH_PREFIX_RESULT EQUAL 0)
	message(FATAL_ERROR "Failed to locate Pytorch installation in Python Environment: ${Python_EXECUTABLE}")
endif()
list(APPEND CMAKE_PREFIX_PATH "${TORCH_INSTALL_PREFIX}/share/cmake/Torch")
unset(CMAKE_CUDA_ARCHITECTURES CACHE)
SET(TORCH_CUDA_ARCH_LIST "8.6")
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
find_package(Torch REQUIRED)
set(HEADERS
	src/csrc/include/utils.h
	src/csrc/include/rmsswiglu.h
	src/csrc/include/rope.h
	src/csrc/include/kv_cache.h
)
set(CUDA_SOURCES
	src/csrc/ops/rmsswiglu.cu
	src/csrc/ops/rope.cu
	src/csrc/ops/kv_cache.cu
)
set(BINDING_SOURCES
	src/csrc/ops/bindings.cpp
)
set(NVCC_FLAGS
	--use_fast_math
	-o3
	--threads=4
	-lineinfo
	-Xcompiler=fPIC
	-D__CUDA_NO_HALF_OPERATORS__
	-D__CUDA_NO_HALF_CONVERSATIONS__
	-D__CUDA_NO_BFLOAT16_CONVERSATIONS__
	-D__CUDA_NO_HALF2_OPERATORS__
)
if(MSVC)
	list(APPEND NVCC_FLAGS -Xcompiler=/wd4819)
else()
	list(APPEND NVCC_FLAGS -Xcompiler=-Wall)
endif()
add_library(llama_pure_cuda SHARED ${CUDA_SOURCES} ${HEADERS})
set_target_properties(llama_pure_cuda PROPERTIES 
	CUDA_ARCHITECTURES "86"
	CUDA_SEPARABLE_COMPILATION ON
	POSITION_INDEPENDENT_CODE ON
)
target_include_directories(llama_pure_cuda PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/include>
	${TORCH_INCLUDE_DIRS}
	${CUDA_INCLUDE_DIRS}
)
target_compile_options(llama_pure_cuda PRIVATE
	$<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}
)
target_link_libraries(llama_pure_cuda PRIVATE
	${TORCH_LIBRARIES}
	${CUDA_LIBRARIES}
	CUDA::cudart
	CUDA::cublas
)
set_target_properties(llama_pure_cuda PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
	POSITION_INDEPENDENT_CODE ON
)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
execute_process(
	COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
	OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
	OUTPUT_STRIP_TRAILING_WHITESPACE
	ERROR_QUIET
)
if(PYBIND11_CMAKE_DIR)
	list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
	find_package(pybind11 CONFIG QUIET)
endif()
if(pybind11_FOUND)
	pybind11_add_module(llama_pure_compute_cuda_ops ${BINDING_SOURCES})
	target_compile_definitions(llama_pure_compute_cuda_ops PRIVATE
		TORCH_EXTENSION_NAME=llama_pure_compute_cuda_ops
	)
	target_link_libraries(llama_pure_compute_cuda_ops PRIVATE
		llama_pure_cuda
		${TORCH_LIBRARIES}
	)
	target_include_directories(llama_pure_compute_cuda_ops PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/include
		${TORCH_INCLUDE_DIRS}
	)
	set_target_properties(llama_pure_compute_cuda_ops PROPERTIES
		LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/llama_pure_compute"
	)
else()
	message(STATUS "pybind11 CMake config not found, using Torch C++ extension flow.")
endif()
message(STATUS "Llama_Pure_Compute CMake Configuration ")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA ToolKit Version: ${CUDAToolkit_VERSION}")
message(STATUS "PYTHON Executable: ${Python3_EXECUTABLE}")
message(STATUS "PyTorch Include Dirs: ${TORCH_INCLUDE_DIRS}")
