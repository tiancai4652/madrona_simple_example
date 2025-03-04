#pragma once

#include <cstring>
#include <cassert>

#include "containers/FixedString.hpp"
#include "containers/FixedVector.hpp"
#include "containers/Pair.hpp"

// CUDA 相关的宏定义
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

