#pragma once

#define SOPHON_VERSION_MAJOR 0
#define SOPHON_VERSION_MINOR 5
#define SOPHON_VERSION_PATCH 0

static_assert(
    SOPHON_VERSION_MINOR < 100,
    "Programming error: you set a minor version that is too big.");
static_assert(
    SOPHON_VERSION_PATCH < 100,
    "Programming error: you set a patch version that is too big.");

#define SOPHON_VERSION                                         \
  (SOPHON_VERSION_MAJOR * 10000 + SOPHON_VERSION_MINOR * 100 +   \
   SOPHON_VERSION_PATCH)

#define SOPHON_USE_CUDA 0
#define SOPHON_USE_NCCL 0
#define SOPHON_USE_ROCM 0
#define SOPHON_USE_RCCL 0
#define SOPHON_USE_REDIS 0
#define SOPHON_USE_IBVERBS 0
#define SOPHON_USE_MPI 0
#define SOPHON_USE_AVX 0
#define SOPHON_USE_LIBUV 0

#define SOPHON_HAVE_TRANSPORT_TCP 1
#define SOPHON_HAVE_TRANSPORT_TCP_TLS 0
#define SOPHON_HAVE_TRANSPORT_IBVERBS 0
#define SOPHON_HAVE_TRANSPORT_UV 0