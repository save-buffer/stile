from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from type_nodes import *

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
    Add = "+"
    Sub = "-"
    Mul = "*"
    Div = "/"
    Repeat = "Repeat"
    Reduce = "Reduce"

@dataclass(frozen=True)
class Enode:
    op : EnodeType
    args : tuple[Any, ...]

class Egraph:
    def __init__(self):
        self.uf = UnionFind()
        self.eclasses : dict[EclassID, set[Enode]] = defaultdict(set)
        self.enodes : dict[Enode, EclassID] = {}
        
        self._rules : list[Callable[[EclassID, Enode], bool]] = [
            self._commutativity,
            self._associativity,
            self._distributivity,
            self._inverse_distributivity,
            self._combine_reductions,
            self._mul_or_div_into_reduction_or_repeat,
            self._mul_or_div_out_of_reduction_or_repeat,
            self._constant_folding,
        ]

    def _commutativity(self, id : EclassID, enode : Enode) -> bool:
        # a + b = b + a
        if enode.op not in (EnodeType.Add, EnodeType.Mul):
            return False
        a, b = enode.args
        swapped_id = self.add(
            Enode(
                op=enode.op,
                args=(b, a)
            )
        )
        if not self.equivalent(id, swapped_id):
            self.merge(id, swapped_id)
            return True
        return False

    def _associativity(self, id : EclassID, enode : Enode) -> bool:
        # (a + b) + c = a + (b + c)
        if enode.op not in (EnodeType.Add, EnodeType.Mul):
            return False
        left, c = enode.args
        left_enodes = self.get_enodes(left)
        for enode_left in left_enodes:
            if enode_left.op == enode.op:
                a, b = enode_left.args
                bc = self.add(
                    Enode(
                        op=enode.op,
                        args=(b, c),
                    )
                )
                reassoc = self.add(
                    Enode(
                        op=enode.op,
                        args=(a, bc),
                    )
                )
                if not self.equivalent(id, reassoc):
                    self.merge(id, reassoc)
                    return True
        return False

    def _distributivity(self, id : EclassID, enode : Enode) -> bool:
        # a * (b + c) = a * b + a * c
        if enode.op != EnodeType.Mul:
            return False
        a, bc = enode.args
        bc_enodes = self.get_enodes(bc)
        for bc_enode in bc_enodes:
            if bc_enode.op == EnodeType.Add:
                b, c = bc_enode.args
                ab = self.add(
                    Enode(
                        op=EnodeType.Mul,
                        args=(a, b),
                    )
                )
                ac = self.add(
                    Enode(
                        op=EnodeType.Mul,
                        args=(a, c),
                    )
                )
                abac = self.add(
                    Enode(
                        op=EnodeType.Add,
                        args=(ab, ac),
                    )
                )
                if not self.equivalent(id, abac):
                    self.merge(id, abac)
                    return True
        return False

    def _inverse_distributivity(self, id : EclassID, enode : Enode) -> bool:
        # a * b + a * c = a * (b + c)
        if enode.op != EnodeType.Add:
            return False
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
                ra, c = ab_enode.args
                if not self.equivalent(la, ra):
                    continue
                bc = self.add(
                    Enode(
                        op=EnodeType.Add,
                        args=(b, c),
                    )
                )
                abc = self.add(
                    Enode(
                        op=EnodeType.Mul,
                        args=(la, bc),
                    )
                )
                if not self.equivalent(id, abc):
                    self.merge(id, abc)
                    return True
        return False

    def _combine_reductions(self, id : EclassID, enode : Enode) -> bool:
        # reduce(x, y) + reduce(y, z) = reduce(x, z)
        if enode.op != EnodeType.Add:
            return False
        lhs, rhs = enode.args
        lhs_enodes = self.get_enodes(lhs)
        rhs_enodes = self.get_enodes(rhs)
        for lhs_enode in lhs_enodes:
            if lhs_enode.op != EnodeType.Reduce:
                continue
            lhs_dim, lhs_child = lhs_enode.args
            for rhs_enode in rhs_enodes:
                if rhs_enode.op != EnodeType.Reduce:
                    continue
                rhs_dim, rhs_child = rhs_enode.args
                if (
                        self.equivalent(lhs_child, rhs_child)
                        and dim_full_dim(lhs_dim) == dim_full_dim(rhs_dim)
                        and dim_end(lhs_dim) == dim_start(rhs_dim)
                ):
                    combined_dim = simplify_dim(
                        Sliced(
                            dim_full_dim(lhs_dim),
                            dim_start(lhs_dim),
                            dim_end(rhs_dim),
                        )
                    )
                    combined = self.add(
                        Enode(
                            op=EnodeType.Reduce,
                            args=(combined_dim, lhs_child),
                        )
                    )
                    if not self.equivalent(id, combined):
                        self.merge(id, combined)
                        return True
        return False

    def _mul_or_div_into_reduction_or_repeat(self, id : EclassID, enode : Enode) -> bool:
        # reduce(x / c) = reduce(x) / c
        if enode.op not in (EnodeType.Mul, EnodeType.Div):
            return False
        lhs, rhs = enode.args
        lhs_enodes = self.get_enodes(lhs)
        rhs_enodes = self.get_enodes(rhs)
        for lhs_enode in lhs_enodes:
            if lhs_enode.op not in (EnodeType.Repeat, EnodeType.Reduce):
                continue

            for rhs_enode in rhs_enodes:
                if rhs_enode.op != EnodeType.Constant:
                    continue

                lhs_dim, lhs_child = lhs_enode.args
                new_inner = self.add(
                    Enode(
                        op=enode.op,
                        args=(lhs_child, rhs),
                    )
                )
                new_outer = self.add(
                    Enode(
                        op=lhs_enode.op,
                        args=(lhs_dim, new_inner),
                    )
                )
                if not self.equivalent(id, new_outer):
                    self.merge(id, new_outer)
                    return True
        return False

    def _mul_or_div_out_of_reduction_or_repeat(self, id : EclassID, enode : Enode) -> bool:
        # reduce(x / c) = reduce(x) / c
        if enode.op not in (EnodeType.Repeat, EnodeType.Reduce):
            return False

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
                new_inner = self.add(
                    Enode(
                        op=enode.op,
                        args=(dim, child_lhs),
                    )
                )
                new_outer = self.add(
                    Enode(
                        op=child_enode.op,
                        args=(new_inner, child_rhs),
                    )
                )
                if not self.equivalent(id, new_outer):
                    self.merge(id, new_outer)
                    return True
        return False

    def _constant_folding(self, id : EclassID, enode : Enode) -> bool:
        if enode.op not in (EnodeType.Add, EnodeType.Sub, EnodeType.Mul, EnodeType.Div):
            return False

        # Make sure we haven't already folded this constant expression
        my_enodes = self.get_enodes(id)
        for my_enode in my_enodes:
            if my_enode.op == EnodeType.Constant:
                return False

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
            return False

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
        new_const = self.add(
            Enode(
                op=EnodeType.Constant,
                args=(new_val,),
            )
        )
        if not self.equivalent(id, new_const):
            self.merge(id, new_const)
            return True
        return False

    def _canonicalize(self, enode : Enode) -> Enode:
        canonicalized_args = tuple(
            arg if not isinstance(arg, EclassID) else self.uf.find(arg)
            for arg in enode.args
        )

        return Enode(
            op=enode.op,
            args=canonicalized_args,
        )

    def _rebuild(self):
        old_enodes = self.enodes
        self.enodes = {}

        for enode, old_id in old_enodes.items():
            canon_enode = self._canonicalize(enode)
            canon_id = self.uf.find(old_id)

            if canon_enode in self.enodes:
                existing_id = self.enodes[canon_enode]
                self.merge(existing_id, canon_id)
            else:
                self.enodes[canon_enode] = canon_id

    def add(self, enode : Enode) -> EclassID:
        enode = self._canonicalize(enode)
        if enode in self.enodes:
            return self.uf.find(self.enodes[enode])

        new_eclass_id = self.uf.make_class()
        self.enodes[enode] = new_eclass_id
        self.eclasses[new_eclass_id].add(enode)
        return new_eclass_id

    def merge(self, x : EclassID, y : EclassID) -> EclassID:
        root_x = self.uf.find(x)
        root_y = self.uf.find(y)

        if root_x == root_y:
            return root_x

        new_root = self.uf.union(root_x, root_y)
        old_root = root_x if new_root == root_y else root_y

        self.eclasses[new_root].update(self.eclasses[old_root])
        del self.eclasses[old_root]

        self._rebuild()
        return new_root

    def get_enodes(self, x : EclassID) -> set[Enode]:
        root = self.uf.find(x)
        return self.eclasses.get(root, set())

    def equivalent(self, x : EclassID, y : EclassID) -> bool:
        return self.uf.find(x) == self.uf.find(y)

    def insert_expression(self, expr : ExprType) -> EclassID:
        match expr:
            case Constant(v):
                enode = Enode(
                    op=EnodeType.Constant,
                    args=(v,),
                )
                return self.add(enode)
            case Tensor(dims):
                enode = Enode(
                    op=EnodeType.Tensor,
                    args=(dims,),
                )
                return self.add(enode)
            case BinaryOp(op, lhs, rhs):
                lhs_id = self.insert_expression(lhs)
                rhs_id = self.insert_expression(rhs)
                enode = Enode(
                    op=EnodeType(op),
                    args=(lhs_id, rhs_id),
                )
                return self.add(enode)
            case Repeat(dim, child):
                child_id = self.insert_expression(child)
                enode = Enode(
                    op=EnodeType.Repeat,
                    args=(dim, child_id),
                )
                return self.add(enode)
            case Reduce(dim, child):
                child_id = self.insert_expression(child)
                enode = Enode(
                    op=EnodeType.Reduce,
                    args=(dim, child_id),
                )
                return self.add(enode)
        assert False, "Unrecognized expression"

    def _apply_rules_to_enode(self, eclass_id : EclassID, enode : Enode) -> bool:
        current_id = self.uf.find(eclass_id)
        for rule in self._rules:
            if rule(current_id, enode):
                return True
        return False

    def apply_rewrites(self, max_iters : int = 10):
        total_merges = 0
        for i in range(max_iters):
            merges_this_iter = 0
            all_enodes = []
            for eclass_id in self.eclasses.keys():
                enodes = self.get_enodes(eclass_id)
                for enode in enodes:
                    all_enodes.append((eclass_id, enode))

            for eclass_id, enode in all_enodes:
                merges_this_iter += 1 if self._apply_rules_to_enode(eclass_id, enode) else 0

            total_merges += merges_this_iter
            if merges_this_iter == 0:
                break

        print(f"Did {total_merges} merges after {i + 1} iters")

