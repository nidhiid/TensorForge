#ifndef TENSORFORGE_TRANSFORMS_PASSES_H
#define TENSORFORGE_TRANSFORMS_PASSES_H

#include <memory>

namespace mlir {
class Pass;
}

namespace tensorforge {

std::unique_ptr<mlir::Pass> createFuseLinearGeluPass();
std::unique_ptr<mlir::Pass> createLowerToLinalgPass();
std::unique_ptr<mlir::Pass> createLowerToCudaRuntimePass();
void registerTensorForgePasses();

} // namespace tensorforge

#endif // TENSORFORGE_TRANSFORMS_PASSES_H
