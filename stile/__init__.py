from .verification import verify_exprs_equivalent
from .specification import parse_spec_into_type
from .type import Type, FullDim, g_dim_registry, Tensor, Constant, TagCond
from .indexing import LoopVariable, AffineExpr, SymbolicIndex, Domain, range_domain

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
    return Tensor(dims=dims, tag=tag)


# --- Rolled loops ---------------------------------------------------------

_active_loop_scopes : list["LoopScope"] = []


class LoopScope:
    """
    Context manager for a rolled loop. Enters the `with` block once with a
    symbolic `LoopVariable` bound to its iteration range; the body traces a
    single parametric instance, and the verifier proves the result holds for
    every integer value of the loop variable in `[start, end)`.
    """
    def __init__(self, name : str, start : SymbolicIndex, end : SymbolicIndex):
        self.var = LoopVariable(name)
        self.lo = start
        self.hi = end
        self.domain = range_domain(self.var, start, end)

    def __enter__(self) -> LoopVariable:
        _active_loop_scopes.append(self)
        return self.var

    def __exit__(self, *_exc):
        popped = _active_loop_scopes.pop()
        assert popped is self


def loop(name : str, start : SymbolicIndex, end : SymbolicIndex) -> LoopScope:
    """
    Create a rolled-loop scope. Use as `with stile.loop("i", 0, N) as i: ...`.
    Inside the block, `i` is a symbolic `LoopVariable` that can participate
    in affine arithmetic (e.g., `i * tile_size`, `i + 1`) to parameterize
    slice bounds and iteration structure.
    """
    return LoopScope(name, start, end)


def active_loop_domain() -> Domain | None:
    """
    The intersection of all currently-active `LoopScope` domains, or `None`
    if no loop is active. Used by the verifier to know what iteration domain
    a traced expression is parametric over.
    """
    if not _active_loop_scopes:
        return None
    variables : set[LoopVariable] = set()
    constraints : set = set()
    for scope in _active_loop_scopes:
        variables |= scope.domain.variables
        constraints |= scope.domain.constraints
    return Domain(frozenset(variables), frozenset(constraints))
