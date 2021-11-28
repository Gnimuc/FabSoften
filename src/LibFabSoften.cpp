#include "fabsoften/LibFabSoften.h"
#include "fabsoften/Beautifier.h"
#include <memory>

using namespace fabsoften;

bool fabsoften_sanity_check(void) { return true; }

FABSOFTEN_LINKAGE fabsoften_context fabsoften_create_context(const char *image,
                                                             const char *model,
                                                             fabsoften_err *err) {
  std::unique_ptr<Beautifier> ptr = std::make_unique<Beautifier>(image, model);

  if (!ptr) {
    fprintf(stderr, "FABSOFTEN ERROR: failed to create `fabsoften::Beautifier`\n");
    *err = fabsoften_error;
  } else {
    *err = fabsoften_success;
  }

  return ptr.release();
}

FABSOFTEN_LINKAGE void fabsoften_dispose(fabsoften_context ctx) {
  delete static_cast<Beautifier *>(ctx);
}

FABSOFTEN_LINKAGE void fabsoften_beautify(fabsoften_context ctx) {
  return static_cast<Beautifier *>(ctx)->soften();
}

FABSOFTEN_LINKAGE void fabsoften_encode(fabsoften_context ctx) {
  auto btf = static_cast<Beautifier *>(ctx);
  btf->encode();
}

FABSOFTEN_LINKAGE size_t fabsoften_get_buffer_size(fabsoften_context ctx) {
  auto btf = static_cast<Beautifier *>(ctx);
  auto &result = btf->getOutputImageBuffer();
  return result.size();
}

FABSOFTEN_LINKAGE void fabsoften_get_data(fabsoften_context ctx, unsigned char *buf) {
  auto btf = static_cast<Beautifier *>(ctx);
  auto &result = btf->getOutputImageBuffer();
  std::copy_n(result.begin(), result.size(), buf);
}
