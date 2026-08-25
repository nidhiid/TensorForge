"""Capture the supported PyTorch MLP and emit TensorForge MLIR."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import fx, nn


class FrontendError(ValueError):
    """Raised when a PyTorch graph is outside TensorForge v0.1's scope."""


@dataclass(frozen=True)
class TensorConstant:
    shape: tuple[int, ...]
    values: tuple[float, ...]

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "TensorConstant":
        value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        return cls(tuple(value.shape), tuple(float(item) for item in value.flatten()))


@dataclass(frozen=True)
class LinearParameters:
    name: str
    weight: TensorConstant
    bias: TensorConstant


@dataclass(frozen=True)
class CapturedMLP:
    """Concrete inputs and parameters extracted from a validated FX graph."""

    graph: fx.GraphModule
    input: TensorConstant
    original_input_shape: tuple[int, ...]
    first_linear: LinearParameters
    second_linear: LinearParameters
    reference_output: TensorConstant

    @property
    def matrix_input_shape(self) -> tuple[int, int]:
        if len(self.original_input_shape) == 2:
            return self.original_input_shape
        batch, sequence, hidden = self.original_input_shape
        return batch * sequence, hidden


def _bias_or_zeros(layer: nn.Linear) -> torch.Tensor:
    if layer.bias is not None:
        return layer.bias
    return torch.zeros(layer.out_features, dtype=torch.float32)


def _linear_parameters(name: str, layer: nn.Linear) -> LinearParameters:
    # PyTorch stores [output, input]. TensorForge MatMul needs [input, output].
    weight = layer.weight.detach().to(device="cpu", dtype=torch.float32)
    return LinearParameters(
        name=name,
        weight=TensorConstant.from_tensor(weight.transpose(0, 1).contiguous()),
        bias=TensorConstant.from_tensor(_bias_or_zeros(layer)),
    )


def _validate_gelu(graph: fx.GraphModule, node: fx.Node) -> None:
    if node.op == "call_module":
        module = graph.get_submodule(str(node.target))
        if not isinstance(module, nn.GELU):
            raise FrontendError("expected GELU between the two Linear layers")
        approximation = module.approximate
    elif node.op == "call_function" and node.target is torch.nn.functional.gelu:
        approximation = node.kwargs.get("approximate", "none")
    else:
        raise FrontendError("expected GELU between the two Linear layers")

    if approximation != "tanh":
        raise FrontendError(
            "GELU must use approximate='tanh' because that is the formula "
            "implemented by TensorForge"
        )


def capture_mlp(model: nn.Module, example_input: torch.Tensor) -> CapturedMLP:
    """Capture and validate exactly Linear -> GELU -> Linear.

    A concrete input supplies the static FP32 shape and values used by the
    current CPU correctness path.
    """

    if not isinstance(example_input, torch.Tensor):
        raise FrontendError("example_input must be a torch.Tensor")
    if example_input.dtype != torch.float32:
        raise FrontendError("TensorForge v0.1 accepts only FP32 input")
    if example_input.dim() not in (2, 3):
        raise FrontendError("input must have shape [M,H] or [B,S,H]")
    if any(dimension <= 0 for dimension in example_input.shape):
        raise FrontendError("all input dimensions must be positive and static")

    model = model.eval()
    graph = fx.symbolic_trace(model)
    nodes = list(graph.graph.nodes)
    placeholders = [node for node in nodes if node.op == "placeholder"]
    outputs = [node for node in nodes if node.op == "output"]
    computation = [
        node for node in nodes if node.op not in ("placeholder", "output")
    ]

    if len(placeholders) != 1 or len(outputs) != 1:
        raise FrontendError("the MLP must have exactly one input and one output")
    if len(computation) != 3:
        raise FrontendError(
            "supported graph must contain exactly Linear -> GELU -> Linear"
        )

    first_node, gelu_node, second_node = computation
    if first_node.op != "call_module" or second_node.op != "call_module":
        raise FrontendError("both Linear operations must be nn.Linear modules")
    first_layer = graph.get_submodule(str(first_node.target))
    second_layer = graph.get_submodule(str(second_node.target))
    if not isinstance(first_layer, nn.Linear) or not isinstance(second_layer, nn.Linear):
        raise FrontendError("supported graph must begin and end with nn.Linear")

    if first_node.args != (placeholders[0],):
        raise FrontendError("the first Linear must directly consume the input")
    if gelu_node.args[0] is not first_node:
        raise FrontendError("GELU must directly consume the first Linear result")
    if second_node.args != (gelu_node,):
        raise FrontendError("the second Linear must directly consume GELU")
    if outputs[0].args[0] is not second_node:
        raise FrontendError("the model output must be the second Linear result")

    _validate_gelu(graph, gelu_node)

    hidden = int(example_input.shape[-1])
    if first_layer.in_features != hidden:
        raise FrontendError(
            f"input hidden size {hidden} does not match the first Linear input "
            f"size {first_layer.in_features}"
        )
    if first_layer.out_features != second_layer.in_features:
        raise FrontendError("the two Linear layers have incompatible dimensions")
    if second_layer.out_features != hidden:
        raise FrontendError(
            "the second Linear output size must restore the input hidden size"
        )

    with torch.no_grad():
        reference = graph(example_input).detach().to(device="cpu", dtype=torch.float32)

    return CapturedMLP(
        graph=graph,
        input=TensorConstant.from_tensor(example_input),
        original_input_shape=tuple(int(size) for size in example_input.shape),
        first_linear=_linear_parameters(str(first_node.target), first_layer),
        second_linear=_linear_parameters(str(second_node.target), second_layer),
        reference_output=TensorConstant.from_tensor(reference),
    )


def _tensor_type(shape: Sequence[int], element_type: str = "f32") -> str:
    dimensions = "x".join(str(dimension) for dimension in shape)
    return f"tensor<{dimensions}x{element_type}>"


def _memref_type(shape: Sequence[int]) -> str:
    dimensions = "x".join(str(dimension) for dimension in shape)
    return f"memref<{dimensions}xf32>"


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        raise FrontendError("non-finite tensor constants are not supported")
    return f"{value:.9e}"


def _format_dense(values: Sequence[float], shape: Sequence[int]) -> str:
    if math.prod(shape) != len(values):
        raise FrontendError("tensor shape does not match its number of values")

    def format_dimension(offset: int, dimensions: Sequence[int]) -> tuple[str, int]:
        if len(dimensions) == 1:
            end = offset + dimensions[0]
            items = ", ".join(_format_float(value) for value in values[offset:end])
            return f"[{items}]", end

        pieces: list[str] = []
        for _ in range(dimensions[0]):
            piece, offset = format_dimension(offset, dimensions[1:])
            pieces.append(piece)
        return f"[{', '.join(pieces)}]", offset

    literal, _ = format_dimension(0, shape)
    return literal


def _constant(name: str, tensor: TensorConstant) -> str:
    return (
        f"    %{name} = arith.constant dense<"
        f"{_format_dense(tensor.values, tensor.shape)}> : "
        f"{_tensor_type(tensor.shape)}"
    )


def emit_mlir(captured: CapturedMLP) -> str:
    """Emit a self-contained TensorForge module for CPU correctness testing."""

    matrix_shape = captured.matrix_input_shape
    intermediate = captured.first_linear.weight.shape[1]
    output_matrix_shape = (matrix_shape[0], captured.second_linear.weight.shape[1])
    original_output_shape = captured.reference_output.shape

    matrix_input = TensorConstant(matrix_shape, captured.input.values)
    lines = [
        "module {",
        "  func.func private @printMemrefF32(memref<*xf32>)",
        "",
        "  func.func @main() attributes {llvm.emit_c_interface} {",
        _constant("input", matrix_input),
        _constant("weight1", captured.first_linear.weight),
        _constant("bias1", captured.first_linear.bias),
        _constant("weight2", captured.second_linear.weight),
        _constant("bias2", captured.second_linear.bias),
        "",
        "    %linear1 = tf.linear %input, %weight1, %bias1",
        f"        : {_tensor_type(matrix_shape)}, "
        f"{_tensor_type(captured.first_linear.weight.shape)}, "
        f"{_tensor_type(captured.first_linear.bias.shape)}",
        f"    %activated = tf.gelu %linear1 : "
        f"{_tensor_type((matrix_shape[0], intermediate))}",
        "    %linear2 = tf.linear %activated, %weight2, %bias2",
        f"        : {_tensor_type((matrix_shape[0], intermediate))}, "
        f"{_tensor_type(captured.second_linear.weight.shape)}, "
        f"{_tensor_type(captured.second_linear.bias.shape)}",
    ]

    output_name = "linear2"
    if tuple(output_matrix_shape) != tuple(original_output_shape):
        shape = ", ".join(str(dimension) for dimension in original_output_shape)
        lines.extend(
            [
                f"    %output = tf.reshape %linear2 to [{shape}]",
                f"        : {_tensor_type(output_matrix_shape)}",
            ]
        )
        output_name = "output"

    output_tensor_type = _tensor_type(original_output_shape)
    output_memref_type = _memref_type(original_output_shape)
    lines.extend(
        [
            f"    %buffer = bufferization.to_buffer %{output_name}",
            f"        : {output_tensor_type} to {output_memref_type}",
            "    %unranked = memref.cast %buffer",
            f"        : {output_memref_type} to memref<*xf32>",
            "    func.call @printMemrefF32(%unranked) : (memref<*xf32>) -> ()",
            "    return",
            "  }",
            "}",
            "",
        ]
    )
    return "\n".join(lines)
