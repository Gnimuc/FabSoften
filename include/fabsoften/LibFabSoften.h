#ifndef LIBFABSOFTEN_H
#define LIBFABSOFTEN_H

#include "fabsoften/Platform.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void *fabsoften_context;

typedef enum { fabsoften_success = 0, fabsoften_error = 1 } fabsoften_err;

FABSOFTEN_LINKAGE bool fabsoften_sanity_check(void);

FABSOFTEN_LINKAGE fabsoften_context fabsoften_create_context(const char *image,
                                                             const char *model,
                                                             fabsoften_err *err);

FABSOFTEN_LINKAGE void fabsoften_dispose(fabsoften_context ctx);

FABSOFTEN_LINKAGE void fabsoften_beautify(fabsoften_context ctx);

FABSOFTEN_LINKAGE void fabsoften_encode(fabsoften_context ctx);

FABSOFTEN_LINKAGE size_t fabsoften_get_buffer_size(fabsoften_context ctx);

FABSOFTEN_LINKAGE void fabsoften_get_data(fabsoften_context ctx, unsigned char *buf);

#ifdef __cplusplus
}
#endif
#endif
