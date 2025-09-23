"""
two mode:
 - infer mode: no start time and end time, only cycles, start and end time will be inferred when gen_profile
 - profile mode: start time and end time already calculated (by perfai) or recorded(by pmu), and can be directly used

"""

from typing import Any, List, Dict, Iterable, Optional, ClassVar, Type, Tuple
from dataclasses import dataclass, field
import uuid
from collections import Counter


@dataclass
class Timestamp:
    # 简单时间戳，单位可为秒或任意抽象tick；本例使用 float 秒
    t: float

    def __lt__(self, other: "Timestamp") -> bool:
        return self.t < other.t

    def __le__(self, other: "Timestamp") -> bool:
        return self.t <= other.t

    def __sub__(self, other: "Timestamp") -> float:
        return self.t - other.t

    def __add__(self, delta: float) -> "Timestamp":
        return Timestamp(self.t + delta)


@dataclass
class SpanContainer:
    CHILD_FIELDS: ClassVar[Tuple[Type["SpanContainer"], ...]] = tuple()

    # 代表一个可度量的持续时间段（如算子在某引擎上的执行）
    name: str = "unknown"
    # 并行结构支持：父子树形关系 + 依赖图（DAG）
    parent_id: Optional[str] = None
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resource_path: List[str] = field(
        default_factory=list
    )  # ["SystemA","TPU0","Engine:matmul"]
    tags: Dict[str, Any] = field(default_factory=dict)
    # children: List["SpanContainer"] = field(default_factory=list)
    deps: List[str] = field(default_factory=list)  # 依赖的其它 span_id 列表

    children: List["SpanContainer"] = field(default_factory=list)

    def add_child(self, *child: "SpanContainer"):
        for c in child:
            assert isinstance(
                c, self.CHILD_FIELDS
            ), f"{c} is not a valid child in {self.name}, support {self.CHILD_FIELDS}"

            c.parent_id = self.span_id
        self.children.extend(child)
        return self

    def add_dep(self, dep_span: "SpanContainer"):
        if dep_span.span_id not in self.deps:
            self.deps.append(dep_span.span_id)

    def walk(self) -> Iterable["SpanContainer"]:
        yield self
        for c in self.children:
            yield from c.walk()

    def time(self) -> float:
        total = 0
        for c in self.children:
            total += c.time()
        return total


@dataclass
class Span(SpanContainer):
    start: Timestamp = field(default_factory=lambda: Timestamp(0))
    end: Timestamp = field(default_factory=lambda: Timestamp(0))

    def add_child(self, *child: "SpanContainer"):
        raise NotImplementedError("Span cannot have children")

    @classmethod
    def from_duration(cls, duration: float):
        return cls(start=Timestamp(0), end=Timestamp(duration))

    def time(self) -> float:
        return self.end.t - self.start.t


@dataclass
class NoiseSpan(Span):
    """任意未知时间"""

    name: str = "noise"


@dataclass
class GdmaSpan(Span):
    name: str = "gdma"


@dataclass
class TiuSpan(Span):
    name: str = "tiu"


@dataclass
class CdmaSpan(Span):
    name: str = "tiu"


@dataclass
class SdmaSpan(Span):
    name: str = "tiu"


@dataclass
class PCIeSpan(Span):
    name: str = "pcie"


@dataclass
class SyncSpan(Span):
    name: str = "sync"


@dataclass
class Pipeline(SpanContainer):
    CHILD_FIELDS = (TiuSpan, GdmaSpan, CdmaSpan, SdmaSpan, NoiseSpan)
    name: str = "pipeline"
    _engines: Dict[str, list[Span]] = field(default_factory=dict)

    def add_child(self, *child: "SpanContainer"):
        ret = super().add_child(*child)
        for c in child:
            self._engines.setdefault(c.name, []).append(c)
        return ret

    def time(self) -> float:
        total = 0
        for engine, spans in self._engines.items():
            engine_total = 0
            for span in spans:
                engine_total += span.time()
            total = max(total, engine_total)
        return total


@dataclass
class Core(SpanContainer):
    """单个 core，一个 core 包含多个 Engine"""

    CHILD_FIELDS = (Pipeline, TiuSpan, GdmaSpan, CdmaSpan, SdmaSpan, NoiseSpan)

    name: str = "core"
    core_id: int = -1

    def add_child(self, *child: "SpanContainer"):
        return super().add_child(*child)


@dataclass
class Chip(SpanContainer):
    """单芯，每芯包含多个 core（TP、EP、DP 级别）"""

    CHILD_FIELDS = (Core, NoiseSpan, CdmaSpan, SyncSpan)

    name: str = "chip"
    chip_id: int = -1

    def add_child(self, *child: "SpanContainer"):
        return super().add_child(*child)

    # def time(self) -> float:
        
        # total = 0
        # core_total = Counter()
        # for c in self.children:
        #     core_total[c.name] += c.time()
        # for name, time in core_total.items():
        #     total = max(total, time)
        # return total


@dataclass
class Card(SpanContainer):
    """单张卡，一张卡包含多个芯"""

    CHILD_FIELDS = (Chip, NoiseSpan, CdmaSpan, PCIeSpan)
    name: str = "card"
    card_id: int = -1

    def add_child(self, *child: "SpanContainer"):
        return super().add_child(*child)


@dataclass
class Panel(SpanContainer):
    """单个计算面板，一个计算面板包含多个 card"""

    CHILD_FIELDS = (Card, NoiseSpan, CdmaSpan, PCIeSpan)
    name: str = "panel"
    panel_id: int = -1

    def add_child(self, *child: "SpanContainer"):
        return super().add_child(*child)


@dataclass
class Cluster(SpanContainer):
    """单个计算集群，一个计算集群包含多个计算面板"""

    CHILD_FIELDS = (Panel, NoiseSpan, PCIeSpan)
    name: str = "cluster"
