# mypy: allow-untyped-defs
import gc
import typing

import torch_tpu

def _dummy_type(name: str) -> type:
    def get_err_fn(is_init: bool):
        def err_fn(obj, *args, **kwargs):
            if is_init:
                class_name = obj.__class__.__name__
            else:
                class_name = obj.__name__
            raise RuntimeError(f"Tried to instantiate dummy base class {class_name}")

        return err_fn

    return type(
        name, (object,), {"__init__": get_err_fn(True), "__new__": get_err_fn(False)}
    )

if not hasattr(torch_tpu._C, "_TPUStreamBase"):
    # Define dummy base classes
    torch_tpu._C.__dict__["_TPUGraph"] = _dummy_type("_TPUGraph")
    torch_tpu._C.__dict__["_graph_pool_handle"] = _dummy_type("_graph_pool_handle")
    torch_tpu._C.__dict__["_tpu_isCurrentStreamCapturing"] = _dummy_type(
        "_cuda_isCurrentStreamCapturing"
    )

from torch_tpu._C import (
    _TPUGraph,
    _graph_pool_handle,
)

# Python shim helps Sphinx process docstrings more reliably.
def graph_pool_handle():
    r"""Return an opaque token representing the id of a graph memory pool.

    See :ref:`Graph memory management<graph-memory-management>`.

    .. warning::
        This API is in beta and may change in future releases.
    """
    return _graph_pool_handle()


# Python shim helps Sphinx process docstrings more reliably.
class TPUGraph(torch_tpu._C._TPUGraph):
    r"""Wrapper around a TPU graph.

    Arguments:
        keep_graph (bool, optional): If ``keep_graph=False``, the
            tpuGraphExec_t will be instantiated on GPU at the end of
            ``capture_end`` and the underlying tpuGraph_t will be
            destroyed. Users who want to query or otherwise modify the
            underlying tpuGraph_t before instantiatiation can set
            ``keep_graph=True`` and access it via ``raw_tpu_graph`` after
            ``capture_end``. Note that the tpuGraphExec_t will not be
            instantiated at the end of ``capture_end`` in this
            case. Instead, it wil be instantiated via an explicit called
            to ``instantiate`` or automatically on the first call to
            ``replay`` if ``instantiate`` was not already called. Calling
            ``instantiate`` manually before ``replay`` is recommended to
            prevent increased latency on the first call to ``replay``. It
            is allowed to modify the raw tpuGraph_t after first calling
            ``instantiate``, but the user must call ``instantiate`` again
            manually to make sure the instantiated graph has these
            changes. Pytorch has no means of tracking these changes.

    .. warning::
        This API is in beta and may change in future releases.

    """

    def __new__(cls, keep_graph=False):
        return super().__new__(cls, keep_graph)

    def capture_begin(self, pool=None):
        r"""Begin capturing TPU work on the current stream.

        Typically, you shouldn't call ``capture_begin`` yourself.
        Use :class:`~torch_tpu.tpu.graph` or :func:`~torch_tpu.tpu.make_graphed_callables`,
        which call ``capture_begin`` internally.

        Arguments:
            pool (optional): Token (returned by :func:`~torch_tpu.tpu.graph_pool_handle` or
                :meth:`other_Graph_instance.pool()<torch_tpu.tpu.TPUGraph.pool>`) that hints this graph may share memory
                with the indicated pool.  See :ref:`Graph memory management<graph-memory-management>`.
        """
        super().capture_begin(pool=pool)

    def capture_end(self):
        r"""End TPU graph capture on the current stream.

        After ``capture_end``, ``replay`` may be called on this instance.

        Typically, you shouldn't call ``capture_end`` yourself.
        Use :class:`~torch_tpu.tpu.graph` or :func:`~torch_tpu.tpu.make_graphed_callables`,
        which call ``capture_end`` internally.
        """
        super().capture_end()

    def instantiate(self):
        r"""Instantiate the TPU graph. Will be called by
        ``capture_end`` if ``keep_graph=False``, or by ``replay`` if
        ``keep_graph=True`` and ``instantiate`` has not already been
        explicitly called. Does not destroy the tpuGraph_t returned
        by ``raw_tpu_graph``.
        """
        super().instantiate()

    def replay(self):
        r"""Replay the TPU work captured by this graph."""
        super().replay()

    def reset(self):
        r"""Delete the graph currently held by this instance."""
        super().reset()

    def pool(self):
        r"""Return an opaque token representing the id of this graph's memory pool.

        This id can optionally be passed to another graph's ``capture_begin``,
        which hints the other graph may share the same memory pool.
        """
        return super().pool()

    def raw_tpu_graph(self):
        r"""Returns the underlying tpuGraph_t. ``keep_graph`` must be True.
        """
        return super().raw_tpu_graph()


class graph:
    r"""Context-manager that captures TPU work into a :class:`torch_tpu.tpu.TPUGraph` object for later replay.

    See :ref:`TPU Graphs <tpu-graph-semantics>` for a general introduction,
    detailed use, and constraints.

    Arguments:
        tpu_graph (torch_tpu.tpu.TPUGraph): Graph object used for capture.
        pool (optional): Opaque token (returned by a call to :func:`~torch_tpu.tpu.graph_pool_handle()` or
            :meth:`other_Graph_instance.pool()<torch_tpu.tpu.TPUGraph.pool>`) hinting this graph's capture
            may share memory from the specified pool. See :ref:`Graph memory management<graph-memory-management>`.
        stream (torch_tpu.tpu.Stream, optional): If supplied, will be set as the current stream in the context.
            If not supplied, ``graph`` sets its own internal side stream as the current stream in the context.

    .. note::
        For effective memory sharing, if you pass a ``pool`` used by a previous capture and the previous capture
        used an explicit ``stream`` argument, you should pass the same ``stream`` argument to this capture.

    .. warning::
        This API is in beta and may change in future releases.

    """

    default_capture_stream: typing.Optional["torch_tpu.Stream"] = None

    def __init__(
        self,
        tpu_graph,
        pool=None,
        stream=None,
    ):
        # Lazy-init of default_capture_stream helps avoid circular-import errors.
        # Not thread safe, but graphs already have the general (explicitly documented)
        # restriction that only one capture may be underway at a time in the process.
        if self.__class__.default_capture_stream is None:
            self.__class__.default_capture_stream = torch_tpu.tpu.current_stream()

        self.pool = () if pool is None else (pool,)
        self.capture_stream = (
            stream if stream is not None else self.__class__.default_capture_stream
        )
        assert self.capture_stream is not None
        self.stream_ctx = torch_tpu.tpu.Stream(self.capture_stream)
        self.tpu_graph = tpu_graph

    def __enter__(self):
        # Free as much memory as we can for the graph
        torch_tpu.tpu.synchronize()
        gc.collect()
        torch_tpu.tpu.empty_cache()

        # Stackoverflow seems comfortable with this pattern
        # https://stackoverflow.com/questions/26635684/calling-enter-and-exit-manually#39172487
        # self.stream_ctx.__enter__()

        self.tpu_graph.capture_begin(*self.pool)

    def __exit__(self, exc_type, exc_value, traceback):
        self.tpu_graph.capture_end()
        # self.stream_ctx.__exit__(exc_type, exc_value, traceback)
        # returning None should propagate exceptions from either capture_end or stream_ctx.__exit__()
