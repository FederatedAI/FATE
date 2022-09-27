from contextlib import contextmanager
from typing import Generator, overload


class Namespace:
    """
    Summary, Metrics may be namespace awared:
    ```
    namespace = Namespace()
    ctx = Context(...summary=XXXSummary(namespace))
    ```
    """

    def __init__(self) -> None:
        self.namespaces = []

    @contextmanager
    def into_subnamespace(self, subnamespace: str):
        self.namespaces.append(subnamespace)
        try:
            yield self
        finally:
            self.namespaces.pop()

    @property
    def namespace(self):
        return ".".join(self.namespaces)

    def fedeation_tag(self) -> str:
        return ".".join(self.namespaces)

    @contextmanager
    def sub_namespace(self, namespace):
        """
        into sub_namespace ``, suffix federation namespace with `namespace`

        Examples:
        ```
        with ctx.sub_namespace("fit"):
            ctx.push(..., trans_key, obj)

        with ctx.sub_namespace("predict"):
            ctx.push(..., trans_key, obj2)
        ```
        `obj1` and `obj2` are pushed with different namespace
        without conflic.
        """

        prev_namespace_state = self._namespace_state

        # into subnamespace
        self._namespace_state = NamespaceState(
            self._namespace_state.sub_namespace(namespace)
        )

        # return sub_ctx
        # ```python
        # with ctx.sub_namespace(xxx) as sub_ctx:
        #     ...
        # ```
        #
        yield self

        # restore namespace state when leaving with context
        self._namespace_state = prev_namespace_state

    @overload
    @contextmanager
    def iter_namespaces(
        self, start: int, stop: int, *, prefix_name=""
    ) -> Generator[Generator["Namespace", None, None], None, None]:
        ...

    @overload
    @contextmanager
    def iter_namespaces(
        self, stop: int, *, prefix_name=""
    ) -> Generator[Generator["Namespace", None, None], None, None]:
        ...

    @contextmanager
    def iter_namespaces(self, *args, prefix_name=""):
        assert 0 < len(args) <= 2, "position argument should be 1 or 2"
        if len(args) == 1:
            start, stop = 0, args[0]
        if len(args) == 2:
            start, stop = args[0], args[1]

        prev_namespace_state = self._namespace_state

        def _state_iterator() -> Generator["Namespace", None, None]:
            for i in range(start, stop):
                # the tags in the iteration need to be distinguishable
                template_formated = f"{prefix_name}iter_{i}"
                self._namespace_state = IterationState(
                    prev_namespace_state.sub_namespace(template_formated)
                )
                yield self

        # with context returns iterator of Contexts
        # namespaec state inside context is changed alone with iterator comsued
        yield _state_iterator()

        # restore namespace state when leaving with context
        self._namespace_state = prev_namespace_state
