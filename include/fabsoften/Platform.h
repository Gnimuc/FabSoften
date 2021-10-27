#ifndef LIBFABSOFTEN_PLATFORM_H
#define LIBFABSOFTEN_PLATFORM_H

#ifndef FABSOFTEN_NO_EXPORTS
#define FABSOFTEN_EXPORTS
#endif
#if defined _WIN32 || defined __CYGWIN__
#ifdef FABSOFTEN_EXPORTS
#ifdef _FABSOFTEN_LIB_NO_EXPORTS_
#define FABSOFTEN_LINKAGE __declspec(dllimport)
#else
#define FABSOFTEN_LINKAGE __declspec(dllexport)
#endif
#endif
#elif defined(FABSOFTEN_EXPORTS) && defined(__GNUC__)
#define FABSOFTEN_LINKAGE __attribute__((visibility("default")))
#endif

#ifndef FABSOFTEN_LINKAGE
#define FABSOFTEN_LINKAGE
#endif

#ifdef __GNUC__
#define FABSOFTEN_DEPRECATED __attribute__((deprecated))
#else
#ifdef _MSC_VER
#define FABSOFTEN_DEPRECATED __declspec(deprecated)
#else
#define FABSOFTEN_DEPRECATED
#endif
#endif

#endif