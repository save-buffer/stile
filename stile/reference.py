"""
Cross-framework reference verification.

The spec-language string (`"2 * X:TTN"`) and a Python *reference function*
written with a typed frontend (`stile.torch`, `stile.jax`) are two ways to
produce the same thing: a stile `ExprType`. `(X * 2).type.et` is structurally
identical to `parse_spec_into_type("2 * X:TTN").et`, and the verifier
(`verify_exprs_equivalent` / `verify_types_equivalent`) compares `ExprType`s —
it never cares where the expected ET came from.

This module is the thin, framework-agnostic glue: run a reference callable on
typed inputs, collect the resulting `Type`(s), and check that the output's
ShapeType and dtype agree with what the kernel declares. The frontends build
the actual typed inputs (array creation is framework-specific); the kernel
frontends (`stile.triton`, `stile.jax.pallas`) feed the resulting `ExprType`
into the same verification path the spec string already uses.
"""

from .type import Type, ShapeType, DataType, dim_full_dim
from .verification import verify_dims_equivalent, verify_dtypes_equivalent


def run_reference(fn, typed_inputs) -> tuple:
    """
    Run a reference `fn` on `typed_inputs` (positional, in declaration
    order) and return its output typed value(s) as a tuple — length 1 for a
    single-output reference, N for one that returns an N-tuple.
    """
    out = fn(*typed_inputs)
    return out if isinstance(out, tuple) else (out,)


def check_output_against_declaration(
    ref_type : Type,
    declared_st : ShapeType,
    declared_dt : "DataType | None",
    label : str,
) -> None:
    """
    Verify a reference's output `Type` matches the kernel's declared
    output shape and dtype. The ExprType is checked separately (against the
    kernel body); here we guard the two things the reference fixes but the
    kernel declares independently — its ShapeType and dtype — so a reference
    that computes the right values into the wrong shape/dtype is rejected.

    `declared_dt` of `None` means the kernel didn't pin a dtype, so dtype is
    not checked (see `verify_dtypes_equivalent`).
    """
    if not verify_dims_equivalent(ref_type.st, declared_st):
        raise ValueError(
            f"{label}: reference produces shape "
            f"{tuple(dim_full_dim(d) for d in ref_type.st)} but the output "
            f"declares {tuple(dim_full_dim(d) for d in declared_st)}."
        )
    if not verify_dtypes_equivalent(ref_type.dt, declared_dt):
        raise ValueError(
            f"{label}: reference produces dtype {ref_type.dt} but the output "
            f"declares {declared_dt}."
        )
