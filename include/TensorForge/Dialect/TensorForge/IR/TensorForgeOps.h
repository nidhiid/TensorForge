#ifndef TENSORFORGE_DIALECT_TENSORFORGE_IR_TENSORFORGEOPS_H
#define TENSORFORGE_DIALECT_TENSORFORGE_IR_TENSORFORGEOPS_H

#include "TensorForge/Dialect/TensorForge/IR/TensorForgeDialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "TensorForge/Dialect/TensorForge/IR/TensorForgeOps.h.inc"

#endif // TENSORFORGE_DIALECT_TENSORFORGE_IR_TENSORFORGEOPS_H
