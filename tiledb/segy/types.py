from typing import TYPE_CHECKING, Dict, List, Tuple, Union

# https://github.com/python/typing/issues/684#issuecomment-548203158
if TYPE_CHECKING:  # pragma: nocover
    from enum import Enum

    class ellipsis(Enum):
        Ellipsis = "..."

    Ellipsis = ellipsis.Ellipsis
    cached_property = property
else:
    ellipsis = type(Ellipsis)
    Ellipsis = Ellipsis
    cached_property = __import__("cached_property").cached_property

Index = Union[int, slice]
ExtendedIndex = Union[int, slice, List[int], ellipsis]
ExtendedIndices = Tuple[ExtendedIndex, ...]

Field = Dict[str, int]
NestedFieldList = Union[List[Field], List[List[Field]], List[List[List[Field]]]]
