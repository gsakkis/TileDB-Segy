# type: ignore

"""Backport of @functools.singledispatchmethod to Python 3.4-3.7."""

import functools

try:
    singledispatchmethod = functools.singledispatchmethod
except AttributeError:

    class singledispatchmethod:
        """Single-dispatch generic method descriptor."""

        def __init__(self, func):
            if not callable(func) and not hasattr(func, "__get__"):
                raise TypeError("{!r} is not callable or a descriptor".format(func))

            self.dispatcher = functools.singledispatch(func)
            self.func = func

        def register(self, cls, method=None):
            return self.dispatcher.register(cls, func=method)

        def __get__(self, obj, cls):
            def _method(*args, **kwargs):
                method = self.dispatcher.dispatch(args[0].__class__)
                return method.__get__(obj, cls)(*args, **kwargs)

            _method.__isabstractmethod__ = self.__isabstractmethod__
            _method.register = self.register
            functools.update_wrapper(_method, self.func)
            return _method

        @property
        def __isabstractmethod__(self):
            return getattr(self.func, "__isabstractmethod__", False)
