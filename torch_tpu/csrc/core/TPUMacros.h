#pragma once


#ifdef _WIN32
#if defined(C10_TPU_BUILD_SHARED_LIBS)
#define C10_TPU_EXPORT __declspec(dllexport)
#define C10_TPU_IMPORT __declspec(dllimport)
#else
#define C10_TPU_EXPORT
#define C10_TPU_IMPORT
#endif
#else // _WIN32
#if defined(__GNUC__)
#define C10_TPU_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define C10_TPU_EXPORT
#endif // defined(__GNUC__)
#define C10_TPU_IMPORT C10_TPU_EXPORT
#endif // _WIN32

// This one is being used by libc10_cuda.so
#ifdef C10_TPU_BUILD_MAIN_LIB
#define C10_TPU_API C10_TPU_EXPORT
#else
#define C10_TPU_API C10_TPU_IMPORT
#endif

/**
 * The maximum number of GPUs that we recognizes.
 */
#define C10_COMPILE_TIME_MAX_TPUS 16