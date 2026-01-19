import math

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .type_nodes import *
from ._rust import RustEgraph, RustExpr # ty: ignore

def expr_type_to_rust_expr(expr : ExprType) -> RustExpr:
    match expr:
        case Constant(v):
            return RustExpr.Constant(v)
        case Tensor(dims):
            return RustExpr.Tensor([dim_name(d) for d in dims])
        case UnaryOp(op, child):
            rust_child = expr_type_to_rust_expr(child)
            match op:
                case "exp":
                    return RustExpr.Exp(rust_child)
                case "sin":
                    return RustExpr.Sin(rust_child)
                case "cos":
                    return RustExpr.Cos(rust_child)
                case "sqrt":
                    return RustExpr.Sqrt(rust_child)
            raise ValueError(f"Unknown unary op {op}")
        case BinaryOp(op, lhs, rhs):
            rust_lhs = expr_type_to_rust_expr(lhs)
            rust_rhs = expr_type_to_rust_expr(rhs)
            match op:
                case "+":
                    return RustExpr.Add(rust_lhs, rust_rhs)
                case "-":
                    return RustExpr.Sub(rust_lhs, rust_rhs)
                case "*":
                    return RustExpr.Mul(rust_lhs, rust_rhs)
                case "/":
                    return RustExpr.Div(rust_lhs, rust_rhs)
                case "max":
                    return RustExpr.BinaryMax(rust_lhs, rust_rhs)
            raise ValueError(f"Unknown binary op {op}")
        case Repeat(dim, child):
            d = dim_name(dim)
            rust_child = expr_type_to_rust_expr(child)
            return RustExpr.Repeat(d, rust_child)
        case Reduce(op, dim, child):
            d = dim_name(dim)
            start, end = dim_start(dim), dim_end(dim)
            rust_child = expr_type_to_rust_expr(child)
            match op:
                case "sum":
                    return RustExpr.Sum(d, start, end, rust_child)
                case "max":
                    return RustExpr.Max(d, start, end, rust_child)
            raise ValueError(f"Unknown reduction op {op}")

def check_if_exprs_equal_rust(x : ExprType, y : ExprType) -> bool:
    egg = RustEgraph()
    x_rust = expr_type_to_rust_expr(x)
    y_rust = expr_type_to_rust_expr(y)
    x_id = egg.insert_expression(x_rust)
    y_id = egg.insert_expression(y_rust)
    return egg.incrementally_check_equivalence(x_id, y_id)

enable_breakpoint = False

def egraph_enable_breakpoint():
    global enable_breakpoint
    enable_breakpoint = True

def maybe_breakpoint():
    global enable_breakpoint
    if enable_breakpoint:
        breakpoint()

@dataclass(frozen=True)
class EclassID:
    id : int

class UnionFind:
    def __init__(self):
        self.parent : list[EclassID] = []
        self.rank : list[int] = []

    def make_class(self) -> EclassID:
        next_id = len(self.parent)
        self.parent.append(EclassID(next_id))
        self.rank.append(0)
        return self.parent[-1]

    def find(self, x : EclassID) -> EclassID:
        while self.parent[x.id] != x:
            self.parent[x.id] = self.parent[self.parent[x.id].id]
            x = self.parent[x.id]
        return x

    def union(self, x : EclassID, y : EclassID) -> EclassID:
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return x

        if self.rank[x.id] < self.rank[y.id]:
            x, y = y, x

        self.parent[y.id] = x
        if self.rank[x.id] == self.rank[y.id]:
            self.rank[x.id] += 1
        return x

class EnodeType(Enum):
    Constant = "Constant"
    Dimension = "Dimension"
    Tensor = "Tensor"
    Exp = "Exp"
    Sin = "Sin"
    Cos = "Cos"
    Sqrt = "Sqrt"
    Add = "+"
    Sub = "-"
    Mul = "*"
    Div = "/"
    BinaryMax = "BinaryMax"
    Repeat = "Repeat"
    Sum = "Sum"
    Max = "Max"

def enode_type_from_unary_op(x : UnaryOpType) -> EnodeType:
    match x:
        case "exp":
            return EnodeType.Exp
        case "sin":
            return EnodeType.Sin
        case "cos":
            return EnodeType.Cos
        case "sqrt":
            return EnodeType.Sqrt

def enode_type_from_binary_op(x : BinaryOpType) -> EnodeType:
    match x:
        case "+" | "-" | "*" | "/":
            return EnodeType(x)
        case "max":
            return EnodeType.BinaryMax

def enode_type_from_reduce_op(x : ReduceOpType) -> EnodeType:
    match x:
        case "sum":
            return EnodeType.Sum
        case "max":
            return EnodeType.Max

@dataclass(frozen=True)
class Enode:
    op : EnodeType
    args : tuple[Any, ...]

class Egraph:
    def __init__(self):
        self.uf = UnionFind()
        self.eclasses : dict[EclassID, dict[Enode, None]] = defaultdict(dict)
        self.enodes : dict[Enode, EclassID] = {}

        self.parents : dict[EclassID, list[tuple[Enode, EclassID]]] = defaultdict(list)
        self.worklist : dict[EclassID, None] = {}

        self.matches : list[tuple[EclassID, Enode | EclassID]] = []

        self._rules : list[Callable[[EclassID, Enode], None]] = [
            self._commutativity,
            self._associativity,
            self._distributivity,
            # self._inverse_distributivity, # This one seems to take an excessive amount of time
            self._div_distributivity,
            self._add_or_subtract_fractions,
            self._multiply_fractions,
            self._decompose_fractions,
            self._identity,
            self._mul_by_zero,
            self._sub_or_div_by_self,
            self._repeat_into_unary_op,
            self._repeat_over_unary_ops,
            self._repeat_over_binary_ops,
            self._combine_reductions,
            self._reorder_reductions,
            self._factor_out_of_sum,
            self._factor_into_sum,
            self._mul_or_div_into_reduction_or_repeat,
            self._mul_or_div_out_of_reduction_or_repeat,
            self._div_of_quotient_to_mul_of_reciprocal,
            self._constant_folding_unary,
            self._constant_folding_binary,
            self._decompose_exp,
            self._recompose_exp,
            self._log_sum_exp_stability,
            self._numerically_stable_softmax,
        ]

    @property
    def dirty(self) -> bool:
        return bool(self.worklist)

    def dump_to_dot(self, file_path : str | None = None):
        if file_path is None:
            file_path = "egraph.dot"

        with open(file_path, 'w') as f:
            f.write("digraph egraph {\n")
            f.write("  compound=true;\n")
            f.write("  node [shape=record];\n\n")

            eclass_to_enodes : dict[EclassID, list[Enode]] = defaultdict(list)
            for enode, eclass_id in self.enodes.items():
                canonical = self.uf.find(eclass_id)
                eclass_to_enodes[canonical].append(enode)

            for eclass_id, enodes in eclass_to_enodes.items():
                f.write(f"  subgraph cluster_{eclass_id.id} {{\n")
                f.write(f"    label=\"eclass {eclass_id.id}\";\n")
                f.write(f"    style=filled;\n")
                f.write(f"    color=lightgrey;\n")

                for i, enode in enumerate(enodes):
                    node_id = f"e{eclass_id.id}_{i}"
                    label = enode.op.value
                    non_eclass_args = [str(arg) for arg in enode.args if not isinstance(arg, EclassID)]
                    if non_eclass_args:
                        label += f"\\n{', '.join(non_eclass_args)}"
                    f.write(f"    {node_id} [label=\"{label}\"];\n")

                f.write("  }\n\n")

            for eclass_id, enodes in eclass_to_enodes.items():
                for i, enode in enumerate(enodes):
                    node_id = f"e{eclass_id.id}_{i}"
                    for arg_idx, arg in enumerate(enode.args):
                        if isinstance(arg, EclassID):
                            target_eclass = self.uf.find(arg)
                            target_enodes = eclass_to_enodes.get(target_eclass, [])
                            if target_enodes:
                                target_node = f"e{target_eclass.id}_0"
                                f.write(f"  {node_id} -> {target_node} [lhead=cluster_{target_eclass.id}];\n")

            f.write("}\n")

    def add_match(self, lhs_id : EclassID, **kwargs):
        if "op" in kwargs or "args" in kwargs:
            assert "op" in kwargs and "args" in kwargs, "Both op and args must be present"
            self.matches.append((lhs_id, Enode(op=kwargs["op"], args=kwargs["args"])))
        elif "id" in kwargs:
            self.matches.append((lhs_id, kwargs["id"]))
        else:
            raise ValueError(f"Invalid {kwargs=}")

    def _commutativity(self, id : EclassID, enode : Enode):
        # a + b = b + a
        if enode.op not in (EnodeType.Add, EnodeType.Mul):
            return
        a, b = enode.args
        self.add_match(
            id,
            op=enode.op,
            args=(b, a)
        )

    def _associativity(self, id : EclassID, enode : Enode):
        # (a + b) + c = a + (b + c)
        if enode.op not in (EnodeType.Add, EnodeType.Mul):
            return
        left, c = enode.args
        left_enodes = self.get_enodes(left)
        for left_enode in left_enodes:
            if left_enode.op != enode.op:
                continue

            a, b = left_enode.args
            bc = Enode(
                op=enode.op,
                args=(b, c),
            )
            self.add_match(
                id,
                op=enode.op,
                args=(a, bc),
            )

    def _distributivity(self, id : EclassID, enode : Enode):
        # a * (b + c) = a * b + a * c
        if enode.op != EnodeType.Mul:
            return
        a, bc = enode.args
        bc_enodes = self.get_enodes(bc)
        for bc_enode in bc_enodes:
            if bc_enode.op != EnodeType.Add:
                continue
            b, c = bc_enode.args
            ab = Enode(
                op=EnodeType.Mul,
                args=(a, b),
            )
            ac = Enode(
                op=EnodeType.Mul,
                args=(a, c),
            )
            self.add_match(
                id,
                op=EnodeType.Add,
                args=(ab, ac),
            )

    def _inverse_distributivity(self, id : EclassID, enode : Enode):
        # a * b + a * c = a * (b + c)
        if enode.op != EnodeType.Add:
            return
        ab, ac = enode.args
        ab_enodes = self.get_enodes(ab)
        ac_enodes = self.get_enodes(ac)
        for ab_enode in ab_enodes:
            if ab_enode.op != EnodeType.Mul:
                continue
            for ac_enode in ac_enodes:
                if ac_enode.op != EnodeType.Mul:
                    continue
                la, b = ab_enode.args
                ra, c = ac_enode.args
                if not self.equivalent(la, ra):
                    continue
                bc = Enode(
                    op=EnodeType.Add,
                    args=(b, c),
                )
                self.add_match(
                    id,
                    op=EnodeType.Mul,
                    args=(la, bc),
                )

    def _div_distributivity(self, id : EclassID, enode : Enode):
        # (a + b) / c = a/c + b/c
        if enode.op != EnodeType.Div:
            return

        num, den = enode.args
        num_enodes = self.get_enodes(num)
        for num_enode in num_enodes:
            if num_enode.op in (EnodeType.Add, EnodeType.Sub):
                a, b = num_enode.args
                ac = Enode(
                    op=EnodeType.Div,
                    args=(a, den),
                )
                bc = Enode(
                    op=EnodeType.Div,
                    args=(b, den)
                )
                self.add_match(
                    id,
                    op=num_enode.op,
                    args=(ac, bc),
                )

    def _repeat_into_unary_op(self, id : EclassID, enode : Enode):
        # f(repeat[D](x)) = repeat[D](f(x))
        if enode.op not in (EnodeType.Exp, EnodeType.Sin, EnodeType.Cos, EnodeType.Sqrt):
            return

        child, = enode.args
        child_enodes = self.get_enodes(child)
        for child_enode in child_enodes:
            if child_enode.op != EnodeType.Repeat:
                continue
            dim, x = child_enode.args
            f_x = Enode(
                op=enode.op,
                args=(x,)
            )
            self.add_match(
                id,
                op=EnodeType.Repeat,
                args=(dim, f_x),
            )

    def _repeat_over_unary_ops(self, id : EclassID, enode : Enode):
        # repeat[D](f(x)) = f(repeat[D](x))
        if enode.op != EnodeType.Repeat:
            return

        dim, child = enode.args
        child_enodes = self.get_enodes(child)
        for child_enode in child_enodes:
            if child_enode.op in (EnodeType.Exp, EnodeType.Sin, EnodeType.Cos, EnodeType.Sqrt):
                inner, = child_enode.args
                inner_repeated = Enode(
                    op=EnodeType.Repeat,
                    args=(dim, inner),
                )
                self.add_match(
                    id,
                    op=child_enode.op,
                    args=(inner_repeated,)
                )
            elif child_enode.op in (EnodeType.Sum, EnodeType.Max):
                inner_dim, inner = child_enode.args
                inner_repeated = Enode(
                    op=EnodeType.Repeat,
                    args=(dim, inner),
                )
                self.add_match(
                    id,
                    op=child_enode.op,
                    args=(inner_dim, inner_repeated),
                )

    def _repeat_over_binary_ops(self, id : EclassID, enode : Enode):
        # repeat[D](a + b) = repeat[D](a) + repeat[D](b)
        if enode.op != EnodeType.Repeat:
            return

        dim, child = enode.args
        child_enodes = self.get_enodes(child)
        for child_enode in child_enodes:
            if child_enode.op in (EnodeType.Add, EnodeType.Sub, EnodeType.Mul, EnodeType.Div, EnodeType.BinaryMax):
                a, b = child_enode.args
                repeat_a = Enode(
                    op=EnodeType.Repeat,
                    args=(dim, a),
                )
                repeat_b = Enode(
                    op=EnodeType.Repeat,
                    args=(dim, b),
                )
                self.add_match(
                    id,
                    op=child_enode.op,
                    args=(repeat_a, repeat_b),
                )

    def _combine_reductions(self, id : EclassID, enode : Enode):
        # sum[x:y] + sum[y:z] = sum(x:z) or binary_max(max[x:y], max[y:z]) = max[x:z]
        match enode.op:
            case EnodeType.Add:
                expected_child_op = EnodeType.Sum
            case EnodeType.BinaryMax:
                expected_child_op = EnodeType.Max
            case _:
                return

        lhs, rhs = enode.args
        lhs_enodes = self.get_enodes(lhs)
        rhs_enodes = self.get_enodes(rhs)
        for lhs_enode in lhs_enodes:
            if lhs_enode.op != expected_child_op:
                continue
            lhs_dim, lhs_child = lhs_enode.args

            for rhs_enode in rhs_enodes:
                if rhs_enode.op != expected_child_op:
                    continue
                rhs_dim, rhs_child = rhs_enode.args
                if (
                    self.equivalent(lhs_child, rhs_child)
                    and dim_full_dim(lhs_dim) == dim_full_dim(rhs_dim)
                    and dim_end(lhs_dim) == dim_start(rhs_dim)
                ):
                    if enode.op == EnodeType.Add:
                        maybe_breakpoint()
                    combined_dim = simplify_dim(
                        Sliced(
                            dim_full_dim(lhs_dim),
                            dim_start(lhs_dim),
                            dim_end(rhs_dim),
                        )
                    )
                    self.add_match(
                        id,
                        op=expected_child_op,
                        args=(combined_dim, lhs_child),
                    )

    def _reorder_reductions(self, id : EclassID, enode : Enode):
        # sum[i](sum[j](f)) = sum[j](sum[i](f)) when dimensions are independent
        if enode.op not in (EnodeType.Sum, EnodeType.Max):
            return

        outer_dim, child = enode.args
        child_enodes = self.get_enodes(child)
        for child_enode in child_enodes:
            if child_enode.op == enode.op:
                inner_dim, inner_child = child_enode.args
                if dim_full_dim(outer_dim) != dim_full_dim(inner_dim):
                    new_inner = Enode(
                        op=enode.op,
                        args=(outer_dim, inner_child),
                    )
                    self.add_match(
                        id,
                        op=enode.op,
                        args=(inner_dim, new_inner),
                    )

    def _factor_out_of_sum(self, id : EclassID, enode : Enode):
        # sum[D](x / repeat[D](y)) = sum[D](x) / y
        if enode.op != EnodeType.Sum:
            return

        sum_dim, child = enode.args
        child_enodes = self.get_enodes(child)
        for child_enode in child_enodes:
            if child_enode.op != EnodeType.Div:
                continue

            lhs, rhs = child_enode.args
            rhs_enodes = self.get_enodes(rhs)
            for rhs_enode in rhs_enodes:
                if rhs_enode.op != EnodeType.Repeat:
                    continue

                repeat_dim, repeat_rhs = rhs_enode.args
                if repeat_dim != dim_full_dim(sum_dim):
                    continue

                sum_x = Enode(
                    op=EnodeType.Sum,
                    args=(sum_dim, lhs),
                )
                self.add_match(
                    id,
                    op=EnodeType.Div,
                    args=(sum_x, repeat_rhs),
                )

    def _factor_into_sum(self, id : EclassID, enode : Enode):
        # sum[D](x) / y = sum[D](x / repeat[D](y))
        if enode.op != EnodeType.Div:
            return

        lhs, rhs = enode.args
        lhs_enodes = self.get_enodes(lhs)
        rhs_enodes = self.get_enodes(rhs)
        for lhs_enode in lhs_enodes:
            if lhs_enode.op != EnodeType.Sum:
                continue

            dim, x = lhs_enode.args
            y_repeated = Enode(
                op=EnodeType.Repeat,
                args=(dim_full_dim(dim), rhs),
            )
            x_div_y = Enode(
                op=EnodeType.Div,
                args=(x, y_repeated),
            )
            self.add_match(
                id,
                op=EnodeType.Sum,
                args=(dim, x_div_y),
            )

    def _mul_or_div_into_reduction_or_repeat(self, id : EclassID, enode : Enode):
        # sum(x / c) = sum(x) / c, or max(x / c) = max(x) / c if c >= 0
        if enode.op not in (EnodeType.Mul, EnodeType.Div):
            return
        lhs, rhs = enode.args
        lhs_enodes = self.get_enodes(lhs)
        rhs_enodes = self.get_enodes(rhs)
        for lhs_enode in lhs_enodes:
            if lhs_enode.op not in (EnodeType.Repeat, EnodeType.Sum, EnodeType.Max):
                continue

            for rhs_enode in rhs_enodes:
                if rhs_enode.op != EnodeType.Constant:
                    continue

                val, = rhs_enode.args
                if lhs_enode.op == EnodeType.Max and val < 0:
                    continue

                lhs_dim, lhs_child = lhs_enode.args
                new_inner = Enode(
                    op=enode.op,
                    args=(lhs_child, rhs),
                )
                self.add_match(
                    id,
                    op=lhs_enode.op,
                    args=(lhs_dim, new_inner),
                )

    def _mul_or_div_out_of_reduction_or_repeat(self, id : EclassID, enode : Enode):
        # sum(x / c) = sum(x) / c, or max(x / c) = max(x) / c if c >= 0
        if enode.op not in (EnodeType.Repeat, EnodeType.Sum, EnodeType.Max):
            return

        dim, child = enode.args
        child_enodes = self.get_enodes(child)
        for child_enode in child_enodes:
            if child_enode.op not in (EnodeType.Mul, EnodeType.Div):
                continue
            child_lhs, child_rhs = child_enode.args
            child_rhs_enodes = self.get_enodes(child_rhs)
            for child_rhs_enode in child_rhs_enodes:
                if child_rhs_enode.op != EnodeType.Constant:
                    continue
                val, = child_rhs_enode.args
                if enode.op == EnodeType.Max and val < 0:
                    continue

                new_inner = Enode(
                    op=enode.op,
                    args=(dim, child_lhs),
                )
                self.add_match(
                    id,
                    op=child_enode.op,
                    args=(new_inner, child_rhs),
                )

    def _mul_by_zero(self, id : EclassID, enode : Enode):
        if enode.op != EnodeType.Mul:
            return

        lhs, rhs = enode.args
        rhs_enodes = self.get_enodes(rhs)
        for rhs_enode in rhs_enodes:
            if rhs_enode.op != EnodeType.Constant:
                continue
            val, = rhs_enode.args
            if val != 0:
                return
            self.add_match(
                id,
                op=EnodeType.Constant,
                args=(0,),
            )

    def _div_of_quotient_to_mul_of_reciprocal(self, id : EclassID, enode : Enode):
        # a / (b / c) = a * (c / b)
        if enode.op != EnodeType.Div:
            return

        lhs, rhs = enode.args
        rhs_enodes = self.get_enodes(rhs)
        for rhs_enode in rhs_enodes:
            if rhs_enode.op != EnodeType.Div:
                continue

            b, c = rhs_enode.args
            reciprocal = Enode(
                op=EnodeType.Div,
                args=(c, b),
            )
            self.add_match(
                id,
                op=EnodeType.Mul,
                args=(lhs, reciprocal),
            )

    def _add_or_subtract_fractions(self, id : EclassID, enode : Enode):
        # (a / d) + (b / d) = (a + b) / d
        if enode.op not in (EnodeType.Add, EnodeType.Sub):
            return

        lhs, rhs = enode.args
        lhs_enodes = self.get_enodes(lhs)
        rhs_enodes = self.get_enodes(rhs)
        for lhs_enode in lhs_enodes:
            if lhs_enode.op != EnodeType.Div:
                continue

            a, lhs_d = lhs_enode.args
            for rhs_enode in rhs_enodes:
                if rhs_enode.op != EnodeType.Div:
                    continue
                b, rhs_d = rhs_enode.args
                if not self.equivalent(lhs_d, rhs_d):
                    continue

                ab = Enode(
                    op=enode.op,
                    args=(a, b),
                )
                self.add_match(
                    id,
                    op=EnodeType.Div,
                    args=(ab, lhs_d),
                )
    
    def _multiply_fractions(self, id : EclassID, enode : Enode):
        # (a / c) * (b / d) = (a * b) / (c * d)
        if enode.op != EnodeType.Mul:
            return

        lhs, rhs = enode.args
        lhs_enodes = self.get_enodes(lhs)
        rhs_enodes = self.get_enodes(rhs)
        for lhs_enode in lhs_enodes:
            if lhs_enode.op != EnodeType.Div:
                continue

            for rhs_enode in rhs_enodes:
                if rhs_enode.op != EnodeType.Div:
                    continue

                a, c = lhs_enode.args
                b, d = rhs_enode.args

                ab = Enode(
                    op=EnodeType.Mul,
                    args=(a, b),
                )
                cd = Enode(
                    op=EnodeType.Mul,
                    args=(c, d),
                )
                self.add_match(
                    id,
                    op=EnodeType.Div,
                    args=(ab, cd),
                )

    def _decompose_fractions(self, id : EclassID, enode : Enode):
        # (a * b) / (c * d) = (a / c) * (b / d)
        if enode.op != EnodeType.Div:
            return

        lhs, rhs = enode.args
        lhs_enodes = self.get_enodes(lhs)
        rhs_enodes = self.get_enodes(rhs)
        for lhs_enode in lhs_enodes:
            if lhs_enode.op != EnodeType.Mul:
                continue
            for rhs_enode in rhs_enodes:
                if rhs_enode.op != EnodeType.Mul:
                    continue
                a, b = lhs_enode.args
                c, d = rhs_enode.args
                ac = Enode(
                    op=EnodeType.Div,
                    args=(a, c),
                )
                bd = Enode(
                    op=EnodeType.Div,
                    args=(b, d),
                )
                self.add_match(
                    id,
                    op=EnodeType.Mul,
                    args=(ac, bd),
                )

    def _sub_or_div_by_self(self, id : EclassID, enode : Enode):
        if enode.op not in (EnodeType.Sub, EnodeType.Div):
            return
        lhs, rhs = enode.args
        if not self.equivalent(lhs, rhs):
            return

        match enode.op:
            case EnodeType.Sub:
                new_val = 0.0
            case EnodeType.Div:
                new_val = 1.0
        self.add_match(
            id,
            op=EnodeType.Constant,
            args=(new_val,),
        )

        
    def _identity(self, id : EclassID, enode : Enode):
        if enode.op not in (EnodeType.Add, EnodeType.Sub, EnodeType.Mul, EnodeType.Div):
            return

        lhs, rhs = enode.args
        rhs_enodes = self.get_enodes(rhs)
        for rhs_enode in rhs_enodes:
            if rhs_enode.op == EnodeType.Constant:
                break
        if rhs_enode.op != EnodeType.Constant:
            return

        val, = rhs_enode.args
        match enode.op:
            case EnodeType.Add | EnodeType.Sub:
                if val != 0:
                    return
            case EnodeType.Mul | EnodeType.Div:
                if val != 1:
                    return
        self.add_match(
            id,
            id=lhs,
        )

    def _constant_folding_unary(self, id : EclassID, enode : Enode):
        if enode.op not in (EnodeType.Exp, EnodeType.Sin, EnodeType.Cos, EnodeType.Sqrt):
            return

        my_enodes = self.get_enodes(id)
        for my_enode in my_enodes:
            if my_enode.op == EnodeType.Constant:
                return

        child, = enode.args
        child_enodes = self.get_enodes(child)
        for child_enode in child_enodes:
            if child_enode.op == EnodeType.Constant:
                break
        if child_enode.op != EnodeType.Constant:
            return

        val, = child_enode.args
        match enode.op:
            case EnodeType.Exp:
                new_val = math.exp(val)
            case EnodeType.Sin:
                new_val = math.sin(val)
            case EnodeType.Cos:
                new_val = math.cos(val)
            case EnodeType.Sqrt:
                new_val = math.sqrt(val)
        new_const = self.add_match(
            id,
            op=EnodeType.Constant,
            args=(new_val,),
        )

    def _constant_folding_binary(self, id : EclassID, enode : Enode):
        if enode.op not in (EnodeType.Add, EnodeType.Sub, EnodeType.Mul, EnodeType.Div):
            return False

        # Make sure we haven't already folded this constant expression
        my_enodes = self.get_enodes(id)
        for my_enode in my_enodes:
            if my_enode.op == EnodeType.Constant:
                return

        lhs, rhs = enode.args
        lhs_enodes = self.get_enodes(lhs)
        rhs_enodes = self.get_enodes(rhs)
        for lhs_enode in lhs_enodes:
            if lhs_enode.op == EnodeType.Constant:
                break
        if lhs_enode.op != EnodeType.Constant:
            return False

        for rhs_enode in rhs_enodes:
            if rhs_enode.op == EnodeType.Constant:
                break
        if rhs_enode.op != EnodeType.Constant:
            return

        val_l, = lhs_enode.args
        val_r, = rhs_enode.args

        match enode.op:
            case EnodeType.Add:
                new_val = val_l + val_r
            case EnodeType.Sub:
                new_val = val_l - val_r
            case EnodeType.Mul:
                new_val = val_l * val_r
            case EnodeType.Div:
                new_val = val_l / val_r
        self.add_match(
            id,
            op=EnodeType.Constant,
            args=(new_val,),
        )

    def _decompose_exp(self, id : EclassID, enode : Enode):
        # exp(x + y) = exp(x) * exp(y) and exp(x - y) = exp(x) / exp(y)
        if enode.op != EnodeType.Exp:
            return
        child, = enode.args
        child_enodes = self.get_enodes(child)
        for child_enode in child_enodes:
            if child_enode.op not in (EnodeType.Add, EnodeType.Sub):
                continue

            lhs, rhs = child_enode.args
            exp_lhs = Enode(
                op=EnodeType.Exp,
                args=(lhs,),
            )
            exp_rhs = Enode(
                op=EnodeType.Exp,
                args=(rhs,),
            )
            match child_enode.op:
                case EnodeType.Add:
                    new_op = EnodeType.Mul
                case EnodeType.Sub:
                    new_op = EnodeType.Div

            self.add_match(
                id,
                op=new_op,
                args=(exp_lhs, exp_rhs),
            )

    def _recompose_exp(self, id : EclassID, enode : Enode):
        # exp(x) * exp(y) = exp(x + y) and exp(x) / exp(y) = exp(x - y)
        if enode.op not in (EnodeType.Mul, EnodeType.Div):
            return

        lhs, rhs = enode.args
        lhs_enodes = self.get_enodes(lhs)
        rhs_enodes = self.get_enodes(rhs)
        for lhs_enode in lhs_enodes:
            if lhs_enode.op != EnodeType.Exp:
                continue
            x, = lhs_enode.args
            for rhs_enode in rhs_enodes:
                if rhs_enode.op != EnodeType.Exp:
                    continue
                y, = rhs_enode.args
                match enode.op:
                    case EnodeType.Mul:
                        combined_op = EnodeType.Add
                    case EnodeType.Div:
                        combined_op = EnodeType.Sub
                xy = Enode(
                    op=combined_op,
                    args=(x, y),
                )
                self.add_match(
                    id,
                    op=EnodeType.Exp,
                    args=(xy,),
                )

    def _log_sum_exp_stability(self, id : EclassID, enode : Enode):
        # sum(exp(x)) = exp(max(x)) * sum(exp(x - max(x)))
        if enode.op != EnodeType.Sum:
            return

        dim, child = enode.args
        child_enodes = self.get_enodes(child)
        for child_enode in child_enodes:
            if child_enode.op == EnodeType.Exp:
                x, = child_enode.args
                # max(x)
                m = Enode(
                    op=EnodeType.Max,
                    args=(dim, x)
                )
                # max(x).repeat()
                m_repeated = Enode(
                    op=EnodeType.Repeat,
                    args=(dim_full_dim(dim), m),
                )
                # x - max(x).repeat()
                x_minus_m = Enode(
                    op=EnodeType.Sub,
                    args=(x, m_repeated),
                )
                # exp(x - max(x).repeat())
                exp_x_minus_m = Enode(
                    op=EnodeType.Exp,
                    args=(x_minus_m,),
                )
                # sum(exp(x - max(x).repeat()))
                sum_exp = Enode(
                    op=EnodeType.Sum,
                    args=(dim, exp_x_minus_m),
                )
                # exp(max(x))
                exp_m = Enode(
                    op=EnodeType.Exp,
                    args=(m,),
                )
                # exp(max(x)) * sum(exp(x - max(x).repeat()))
                self.add_match(
                    id,
                    op=EnodeType.Mul,
                    args=(exp_m, sum_exp),
                )

    def _numerically_stable_softmax(self, id : EclassID, enode : Enode):
        # exp(x) / sum(exp(x)).repeat() = exp(x - max(x).repeat()) / sum(exp(x - max(x).repeat()))
        if enode.op != EnodeType.Div:
            return

        lhs, rhs = enode.args
        lhs_enodes = self.get_enodes(lhs)
        rhs_enodes = self.get_enodes(rhs)
        for lhs_enode in lhs_enodes:
            if lhs_enode.op != EnodeType.Exp:
                continue
            for rhs_enode in rhs_enodes:
                if rhs_enode.op != EnodeType.Repeat:
                    continue
                repeat_dim, maybe_sum = rhs_enode.args
                maybe_sum_enodes = self.get_enodes(maybe_sum)
                for maybe_sum_enode in maybe_sum_enodes:
                    if maybe_sum_enode.op != EnodeType.Sum:
                        continue

                    sum_dim, maybe_lhs = maybe_sum_enode.args
                    if not self.equivalent(lhs, maybe_lhs):
                        continue
                    if dim_full_dim(sum_dim) != repeat_dim:
                        continue

                    x, = lhs_enode.args
                    max_x = Enode(
                        op=EnodeType.Max,
                        args=(sum_dim, x),
                    )
                    max_x_repeated = Enode(
                        op=EnodeType.Repeat,
                        args=(repeat_dim, max_x),
                    )
                    x_corrected = Enode(
                        op=EnodeType.Sub,
                        args=(x, max_x_repeated),
                    )
                    exp_x_corrected = Enode(
                        op=EnodeType.Exp,
                        args=(x_corrected,),
                    )
                    sum_exp_x_corrected = Enode(
                        op=EnodeType.Sum,
                        args=(sum_dim, exp_x_corrected),
                    )
                    sum_exp_x_corrected_repeated = Enode(
                        op=EnodeType.Repeat,
                        args=(repeat_dim, sum_exp_x_corrected),
                    )
                    self.add_match(
                        id,
                        op=EnodeType.Div,
                        args=(exp_x_corrected, sum_exp_x_corrected_repeated),
                    )
        

    def _canonicalize(self, enode : Enode) -> Enode:
        canonicalized_args = tuple(
            arg if not isinstance(arg, EclassID) else self.uf.find(arg)
            for arg in enode.args
        )

        return Enode(
            op=enode.op,
            args=canonicalized_args,
        )

    def _recursively_insert_enode(self, enode : Enode) -> EclassID:
        inserted_args = tuple(
            arg if not isinstance(arg, Enode) else self._recursively_insert_enode(arg)
            for arg in enode.args
        )
        return self.add(
            Enode(
                op=enode.op,
                args=inserted_args,
            )
        )

    def add(self, enode : Enode) -> EclassID:
        enode = self._canonicalize(enode)
        if enode in self.enodes:
            return self.uf.find(self.enodes[enode])

        new_eclass_id = self.uf.make_class()
        self.eclasses[new_eclass_id][enode] = None

        for child in enode.args:
            if isinstance(child, EclassID):
                self.parents[child].append((enode, new_eclass_id))

        self.enodes[enode] = new_eclass_id
        return new_eclass_id

    def merge(self, x : EclassID, y : EclassID) -> EclassID:
        root_x = self.uf.find(x)
        root_y = self.uf.find(y)

        if root_x == root_y:
            return root_x

        new_root = self.uf.union(root_x, root_y)
        old_root = root_x if root_x != new_root else root_y
        self.eclasses[new_root].update(self.eclasses[old_root])
        del self.eclasses[old_root]

        self.parents[new_root].extend(self.parents[old_root])
        del self.parents[old_root]

        self.worklist[new_root] = None
        return new_root

    def rebuild(self):
        while self.worklist:
            todo : dict[EclassID, None] = { self.uf.find(x) : None for x in self.worklist }
            self.worklist.clear()
            for eclass in todo:
                self._repair(eclass)

    def _repair(self, eclass : EclassID):
        for parent_enode, parent_eclass in self.parents[eclass]:
            self.enodes.pop(parent_enode, None)
            parent_enode = self._canonicalize(parent_enode)
            self.enodes[parent_enode] = self.uf.find(parent_eclass)

        new_parents : dict[Enode, EclassID] = {}
        for parent_enode, parent_eclass in self.parents[eclass]:
            parent_enode = self._canonicalize(parent_enode)
            if parent_enode in new_parents:
                self.merge(parent_eclass, new_parents[parent_enode])
            new_parents[parent_enode] = self.uf.find(parent_eclass)
        self.parents[eclass] = list(new_parents.items())

    def get_enodes(self, x : EclassID) -> dict[Enode, None]:
        root = self.uf.find(x)
        return self.eclasses.get(root, dict())

    def equivalent(self, x : EclassID, y : EclassID) -> bool:
        if self.dirty:
            raise RuntimeError("Tried to query dirty egraph - be sure to rebuild() after modifications")
        return self.uf.find(x) == self.uf.find(y)

    def _insert_expression_norebuild(self, expr : ExprType) -> EclassID:
        match expr:
            case Constant(v):
                enode = Enode(
                    op=EnodeType.Constant,
                    args=(v,),
                )
                return self.add(enode)
            case Tensor(dims):
                canon_dims = tuple(dim_full_dim(d) for d in dims)
                enode = Enode(
                    op=EnodeType.Tensor,
                    args=canon_dims,
                )
                return self.add(enode)
            case UnaryOp(op, child):
                child_id = self.insert_expression(child)
                enode = Enode(
                    op=enode_type_from_unary_op(op),
                    args=(child_id,),
                )
                return self.add(enode)
            case BinaryOp(op, lhs, rhs):
                lhs_id = self.insert_expression(lhs)
                rhs_id = self.insert_expression(rhs)
                enode = Enode(
                    op=enode_type_from_binary_op(op),
                    args=(lhs_id, rhs_id),
                )
                return self.add(enode)
            case Repeat(dim, child):
                child_id = self.insert_expression(child)
                enode = Enode(
                    op=EnodeType.Repeat,
                    args=(dim_full_dim(dim), child_id),
                )
                return self.add(enode)
            case Reduce(op, dim, child):
                child_id = self.insert_expression(child)
                enode = Enode(
                    op=enode_type_from_reduce_op(op),
                    args=(dim, child_id),
                )
                return self.add(enode)
        assert False, "Unrecognized expression"

    def insert_expression(self, expr : ExprType) -> EclassID:
        result = self._insert_expression_norebuild(expr)
        self.rebuild()
        return result

    def _apply_rules_to_enode(self, eclass_id : EclassID, enode : Enode):
        current_id = self.uf.find(eclass_id)
        for rule in self._rules:
            rule(current_id, enode)

    def apply_rewrites(self) -> int:
        self.matches.clear()
        for eclass_id in self.eclasses.keys():
            enodes = self.get_enodes(eclass_id)
            for enode in enodes:
                self._apply_rules_to_enode(eclass_id, enode)

        for matched_id, match in self.matches:
            if isinstance(match, Enode):
                id_to_merge = self._recursively_insert_enode(match)
            else:
                assert isinstance(match, EclassID)
                id_to_merge = match
            self.merge(matched_id, id_to_merge)

        self.rebuild()

        return len(self.matches)

    def incrementally_check_equivalence(self, x : EclassID, y : EclassID, max_iters : int) -> bool:
        total_merges = 0
        for i in range(max_iters):
            merges_this_iter = self.apply_rewrites()
            print(f"Iter {i} did {merges_this_iter} merges, {len(self.enodes)} enodes")            
            total_merges += merges_this_iter
            
            if self.equivalent(x, y):
                return True
        return False
        
    def perform_equality_saturation(self, max_iters : int):
        total_merges = 0
        for i in range(max_iters):
            merges_this_iter = self.apply_rewrites()
            print(f"Iter {i} did {merges_this_iter} merges, {len(self.enodes)} enodes")
            total_merges += merges_this_iter
            if merges_this_iter == 0:
                break

