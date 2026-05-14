from .verification import verify_exprs_equivalent
from .specification import parse_spec_into_type
from .type import (
    Type, FullDim, g_dim_registry, Tensor, Constant, TagCond,
    _reset_tensor_counter,
)
from .indexing import (
    SymbolicInt, LoopVariable, AffineExpr, SymbolicIndex, Domain, range_domain,
    LoopScope, loop, active_loop_domain, _active_loop_scopes,
    RuntimeScalar, runtime_scalar, _g_runtime_scalars,
    SymInfo, symint_info, _g_symint_metadata,
    tensor_element,
    declare_index_properties, index_has_property, _g_index_properties,
    declare_block_pairing, paired_index_for_offsets, _g_block_pairings,
    declare_tensor_boundary, tensor_boundary, resolve_symbolic_index,
    _g_tensor_boundaries,
)
from .jax._core import _g_runtime_arrs

def dim(name : str, size : int) -> FullDim:
    return FullDim(name, size)

def expr_simplifies(
    expr : Type,
    spec : str,
) -> bool:
    spec_type = parse_spec_into_type(spec)
    return verify_exprs_equivalent(expr.type.et, spec_type.et)

def reset_stile():
    g_dim_registry.clear()
    _active_loop_scopes.clear()
    _g_runtime_scalars.clear()
    _g_symint_metadata.clear()
    _g_index_properties.clear()
    _g_block_pairings.clear()
    _g_tensor_boundaries.clear()
    _g_runtime_arrs.clear()
    _reset_tensor_counter()


def mask_expr(
    dims : tuple[FullDim, ...],
    domain : Domain,
) -> Tensor:
    """
    A tagged Tensor whose value is `1` on positions in `domain` and `0`
    elsewhere. The tag is `Cond(domain, Value(1), Value(0))`. `domain`'s
    constraints should reference `LoopVariable`s named after the tensor's
    dims — those are the symbolic dim-indices.

    Common masks (causal, band, block-diagonal) are library wrappers over
    this primitive, produced by constructing the appropriate `Domain`.
    """
    tag = TagCond(
        domain=domain,
        if_true=Constant(1.0),
        if_false=Constant(0.0),
    )
    # Mask tensors use a fixed name so two `mask_expr` calls with the
    # same dims+predicate produce equal tensors (the tag carries the
    # identifying info).
    return Tensor(dims=dims, tag=tag, name="_mask")


