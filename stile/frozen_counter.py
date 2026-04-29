from dataclasses import dataclass
from typing import Generic, Iterable, Iterator, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class FrozenCounter(Generic[T]):
    """Immutable, hashable multiset. Zero counts are never stored.

    Backing store is a frozenset of (key, count) pairs so equality and hashing
    come for free from the dataclass. Construct via `from_iterable`, `from_dict`,
    or `empty` — not by passing the frozenset directly unless it's already
    been sanitized.
    """
    _items : frozenset[tuple[T, int]]

    def __hash__(self) -> int:
        cached = getattr(self, '_h', None)
        if cached is not None:
            return cached
        h = hash(self._items)
        object.__setattr__(self, '_h', h)
        return h

    @staticmethod
    def empty() -> "FrozenCounter[T]":
        return _EMPTY

    @staticmethod
    def from_iterable(items : Iterable[T]) -> "FrozenCounter[T]":
        counts : dict[T, int] = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        if not counts:
            return _EMPTY
        return FrozenCounter(frozenset(counts.items()))

    @staticmethod
    def from_dict(counts : dict[T, int]) -> "FrozenCounter[T]":
        if not counts:
            return _EMPTY
        items = [(k, v) for k, v in counts.items() if v != 0]
        if not items:
            return _EMPTY
        return FrozenCounter(frozenset(items))

    def __getitem__(self, key : T) -> int:
        for k, v in self._items:
            if k == key:
                return v
        return 0

    def __iter__(self) -> Iterator[T]:
        for k, _ in self._items:
            yield k

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __contains__(self, key : object) -> bool:
        return any(k == key for k, _ in self._items)

    def items(self) -> Iterator[tuple[T, int]]:
        yield from self._items

    def keys(self) -> Iterator[T]:
        for k, _ in self._items:
            yield k

    def values(self) -> Iterator[int]:
        for _, v in self._items:
            yield v

    def total(self) -> int:
        return sum(v for _, v in self._items)

    def __add__(self, other : "FrozenCounter[T]") -> "FrozenCounter[T]":
        """Multiset sum: counts add."""
        if not other._items:
            return self
        if not self._items:
            return other
        result : dict[T, int] = {}
        for k, v in self._items:
            result[k] = result.get(k, 0) + v
        for k, v in other._items:
            result[k] = result.get(k, 0) + v
        return FrozenCounter.from_dict(result)

    def __sub__(self, other : "FrozenCounter[T]") -> "FrozenCounter[T]":
        """Multiset difference: counts subtract, clamped at zero."""
        if not other._items:
            return self
        if not self._items:
            return self
        result : dict[T, int] = dict(self._items)
        for k, v in other._items:
            remaining = result.get(k, 0) - v
            if remaining > 0:
                result[k] = remaining
            else:
                result.pop(k, None)
        return FrozenCounter(frozenset(result.items()))

    def __and__(self, other : "FrozenCounter[T]") -> "FrozenCounter[T]":
        """Multiset intersection: min of counts. Useful for GCD-style cancellation."""
        if not self._items or not other._items:
            return FrozenCounter(frozenset())
        lhs : dict[T, int] = dict(self._items)
        result : dict[T, int] = {}
        for k, v in other._items:
            if k in lhs:
                result[k] = min(v, lhs[k])
        return FrozenCounter(frozenset(result.items()))

    def __or__(self, other : "FrozenCounter[T]") -> "FrozenCounter[T]":
        """Multiset union: max of counts."""
        result : dict[T, int] = dict(self._items)
        for k, v in other._items:
            result[k] = max(result.get(k, 0), v)
        return FrozenCounter(frozenset(result.items()))

    def map(self, f) -> "FrozenCounter":
        return FrozenCounter.from_dict({ f(k) : v for k, v in self._items })

    def __repr__(self) -> str:
        body = ", ".join(f"{k!r}: {v}" for k, v in sorted(self._items, key=lambda kv: repr(kv[0])))
        return f"FrozenCounter({{{body}}})"


_EMPTY : FrozenCounter = FrozenCounter(frozenset())
