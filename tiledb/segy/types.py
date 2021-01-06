from typing import TYPE_CHECKING, Dict, List, Union

import numpy as np

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
    from cached_property import cached_property  # noqa: F401

Int = Union[int, np.integer]
Index = Union[Int, slice]
Field = Dict[str, int]
NestedFieldList = Union[List[Field], List[List[Field]], List[List[List[Field]]]]
