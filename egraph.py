from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from type_nodes import *

@dataclass(frozen=True)
class EClassID:
    id : int

class UnionFind:
    def __init__(self):
        self.parent : list[EClassID] = []
        self.rank : list[int] = []

    def make_class(self) -> EClassID:
        next_id = len(self.parent)
        self.parent.append(EClassID(next_id))
        self.rank.append(0)
        return self.parent[-1]

    def find(self, x : EClassID) -> EClassID:
        while self.parent[x.id] != x:
            self.parent[x.id] = self.parent[self.parent[x.id].id]
            x = self.parent[x.id]
        return x

    def union(self, x : EClassID, y : EClassID) -> EClassID:
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

class ENodeType(Enum):
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
class ENode:
    op : ENodeType
    args : tuple[Any, ...]

class EGraph:
    def __init__(self):
        self.uf = UnionFind()
        self.eclasses : dict[EClassID, set[ENode]] = defaultdict(set)
        self.enodes : dict[ENode, EClassID] = {}
        
        self._rules : list[Callable[[EClassID, ENode], bool]] = [
            self._commutativity,
            self._associativity,
            self._combine_reductions,
        ]

    def _commutativity(self, id : EClassID, enode : ENode) -> bool:
        if enode.op == ENodeType.Add or enode.op == ENodeType.Mul:
            a, b = enode.args
            swapped_id = self.add(
                ENode(
                    op=enode.op,
                    args=(b, a)
                )
            )
            if not self.equivalent(id, swapped_id):
                self.merge(id, swapped_id)
                return True
        return False

    def _associativity(self, id : EClassID, enode : ENode) -> bool:
        # (a + b) + c = a + (b + c)
        if enode.op == ENodeType.Add or enode.op == ENodeType.Mul:
            left, c = enode.args
            left_enodes = self.get_enodes(left)
            for enode_left in left_enodes:
                if enode_left.op == enode.op:
                    a, b = enode_left.args
                    bc = self.add(
                        ENode(
                            op=enode.op,
                            args=(b, c),
                        )
                    )
                    reassoc = self.add(
                        ENode(
                            op=enode.op,
                            args=(a, bc),
                        )
                    )
                    if not self.equivalent(id, reassoc):
                        self.merge(id, reassoc)
                        return True
        return False

    def _combine_reductions(self, id : EClassID, enode : ENode) -> bool:
        if enode.op == ENodeType.Add:
            lhs, rhs = enode.args
            lhs_enodes = self.get_enodes(lhs)
            rhs_enodes = self.get_enodes(rhs)
            for lhs_enode in lhs_enodes:
                if lhs_enode.op != ENodeType.Reduce:
                    continue
                lhs_dim, lhs_child = lhs_enode.args
                for rhs_enode in rhs_enodes:
                    if rhs_enode.op != ENodeType.Reduce:
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
                            ENode(
                                op=ENodeType.Reduce,
                                args=(combined_dim, lhs_child),
                            )
                        )
                        if not self.equivalent(id, combined):
                            self.merge(id, combined)
                            return True
        return False


    def _canonicalize(self, enode : ENode) -> ENode:
        canonicalized_args = tuple(
            arg if not isinstance(arg, EClassID) else self.uf.find(arg)
            for arg in enode.args
        )

        return ENode(
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

    def add(self, enode : ENode) -> EClassID:
        enode = self._canonicalize(enode)
        if enode in self.enodes:
            return self.uf.find(self.enodes[enode])

        new_eclass_id = self.uf.make_class()
        self.enodes[enode] = new_eclass_id
        self.eclasses[new_eclass_id].add(enode)
        return new_eclass_id

    def merge(self, x : EClassID, y : EClassID) -> EClassID:
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

    def get_enodes(self, x : EClassID) -> set[ENode]:
        root = self.uf.find(x)
        return self.eclasses.get(root, set())

    def equivalent(self, x : EClassID, y : EClassID) -> bool:
        return self.uf.find(x) == self.uf.find(y)

    def insert_expression(self, expr : ENodeType) -> EClassID:
        match expr:
            case Constant(v):
                enode = ENode(
                    op=ENodeType.Constant,
                    args=(v,),
                )
                return self.add(enode)
            case Tensor(dims):
                enode = ENode(
                    op=ENodeType.Tensor,
                    args=(dims,),
                )
                return self.add(enode)
            case BinaryOp(op, lhs, rhs):
                lhs_id = self.insert_expression(lhs)
                rhs_id = self.insert_expression(rhs)
                enode = ENode(
                    op=ENodeType(op),
                    args=(lhs_id, rhs_id),
                )
                return self.add(enode)
            case Repeat(dim, child):
                child_id = self.insert_expression(child)
                enode = ENode(
                    op=ENodeType.Repeat,
                    args=(dim, child_id),
                )
                return self.add(enode)
            case Reduce(dim, child):
                child_id = self.insert_expression(child)
                enode = ENode(
                    op=ENodeType.Reduce,
                    args=(dim, child_id),
                )
                return self.add(enode)
        assert False, "Unrecognized expression"

    def _apply_rules_to_enode(self, eclass_id : EClassID, enode : ENode) -> bool:
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

        print(f"Did {total_merges} merges after {max_iters} iters")

