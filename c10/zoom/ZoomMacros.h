#pragma once

// See c10/macros/Export.h for a detailed explanation of what the function
// of these macros are.  We need one set of macros for every separate library
// we build.

#ifdef _WIN32
#if defined(C10_HIP_BUILD_SHARED_LIBS)
#define C10_ZOOM_EXPORT __declspec(dllexport)
#define C10_ZOOM_IMPORT __declspec(dllimport)
#else
#define C10_ZOOM_EXPORT
#define C10_ZOOM_IMPORT
#endif
#else // _WIN32
#if defined(__GNUC__)
#define C10_ZOOM_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define C10_ZOOM_EXPORT
#endif // defined(__GNUC__)
#define C10_ZOOM_IMPORT C10_ZOOM_EXPORT
#endif // _WIN32

// This one is being used by libc10_zoom.so
#ifdef C10_ZOOM_BUILD_MAIN_LIB
#define C10_ZOOM_API C10_ZOOM_EXPORT
#else
#define C10_ZOOM_API C10_ZOOM_IMPORT
#endif

/**
 * The maximum number of GPUs that we recognizes. Increasing this beyond the
 * initial limit of 16 broke Caffe2 testing, hence the ifdef guards.
 * This value cannot be more than 128 because our DeviceIndex is a uint8_t.
o */
#ifdef FBCODE_CAFFE2
// fbcode depends on this value being 16
#define C10_COMPILE_TIME_MAX_GPUS 16
#else
#define C10_COMPILE_TIME_MAX_GPUS 120
#endif