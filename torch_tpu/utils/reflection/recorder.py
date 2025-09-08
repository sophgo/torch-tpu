import torch
from .timestamp import Timestamp
from typing import Callable
import numpy as np
from torch import Tensor
from typing import Union, List, Dict, Any, Optional, Set
import uuid
from enum import Enum

# 全局计算图栈，用于处理嵌套的计算图
graph_stack = []
current_graph = None


class NodeType(Enum):
    """节点类型枚举"""

    VALUE = "value"  # Tensor 值节点
    OPERATION = "operation"  # 计算操作节点
    ATTR = "attr"  # 属性节点


class EdgeType(Enum):
    """边类型枚举"""

    OPERAND = "operand"  # 操作数（输入）
    RESULT = "result"  # 结果（输出）


class Node:
    """计算图节点基类"""

    def __init__(self, name: str = None, node_type: NodeType = None):
        self.name = name
        self.node_id = str(uuid.uuid4())
        self.node_type = node_type
        self.incoming_edges: List[UseEdge] = []
        self.outgoing_edges: List[UseEdge] = []
        self.metadata: Dict[str, Any] = {}

    def add_incoming_edge(self, edge: "UseEdge"):
        """添加输入边"""
        self.incoming_edges.append(edge)

    def add_outgoing_edge(self, edge: "UseEdge"):
        """添加输出边"""
        self.outgoing_edges.append(edge)

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.node_id}, type={self.node_type})"


class Value(Node):
    """值节点，表示 Tensor 数据"""

    def __init__(self, tensor_data: torch.Tensor, name: str = None):
        super().__init__(name, NodeType.VALUE)
        self.tensor_data = tensor_data
        self.shape = tuple(tensor_data.shape)
        self.dtype = tensor_data.dtype

    def __repr__(self):
        shape_str = f"shape={self.shape} " if self.shape else ""
        dtype_str = f"dtype={self.dtype} " if self.dtype else ""
        return f"{self.name}: Tensor[{shape_str}{dtype_str}]"


class Attr(Node):
    def __init__(self, attr: Any, name: str = None):
        super().__init__(name, NodeType.ATTR)
        self.attr = attr

    def __repr__(self):
        if self.attr is None:
            return f"{self.name} = None"
        return f"{self.name}: {self.attr.__class__.__name__} = {self.attr}"


class ValueAttr(Node):
    def __init__(self, attr: Any, name: str = None):
        super().__init__(name, NodeType.ATTR)
        self.attr = attr
        self.shape = getattr(attr, "shape", None) if attr is not None else None
        self.dtype = getattr(attr, "dtype", None) if attr is not None else None

    def __repr__(self):
        shape_str = f"shape={self.shape}" if self.shape else ""
        dtype_str = f"dtype={self.dtype}" if self.dtype else ""
        return f"{self.name}: {self.attr.__class__.__name__}[{shape_str}{dtype_str}]"


class Operation(Node):
    """操作节点，表示计算操作"""

    @classmethod
    def from_function(cls, operation_func, args, kwargs):
        """
        优先从子类中 retrieve Operation 的子类，如果没有则创建新的 Operation 子类
        """
        for sub_class in cls.sub_classes.values():
            if sub_class.is_target_class(operation_func, args, kwargs):
                return sub_class(operation_func)
        return cls(operation_func)

    @classmethod
    def is_target_class(cls, operation_func, args, kwargs):

        if cls.target_function == operation_func:
            return True
        if cls.target_funcname == operation_func.__name__:
            return True

        return cls.__name__ == operation_func.__name__

    sub_classes: Dict[str, "Operation"] = {}

    target_function: Callable = None
    target_funcname: str = None

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.sub_classes[cls.__name__] = cls

    def __init__(
        self,
        operation_func,
        args: tuple = None,
        kwargs: dict = None,
        node_id: str = None,
    ):
        super().__init__(node_id, NodeType.OPERATION)
        self.operation_func = operation_func
        self.operation_name = getattr(operation_func, "__name__", "unknown")
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.result_nodes: List[Value] = []

    def add_result(self, result_node: Value):
        """添加结果节点"""
        self.result_nodes.append(result_node)

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.node_id}, operation={self.operation_name})"

    def get_timing_profile(self) -> List[Union[float, Timestamp]]:
        return []


class SetItem(Operation):
    target_function = torch.Tensor.__setitem__


class GetItem(Operation):
    target_function = torch.Tensor.__getitem__


class UseEdge:
    """计算图边，连接节点"""

    def __init__(self, source: Node, target: Node, edge_type: EdgeType):
        self.edge_id = str(uuid.uuid4())
        self.source = source
        self.target = target
        self.edge_type = edge_type
        self.metadata: Dict[str, Any] = {}

    def __repr__(self):
        return f"Edge(id={self.edge_id}, {self.source.node_id} -> {self.target.node_id}, type={self.edge_type})"


class Operand(UseEdge):
    def __init__(self, source: Node, target: Node, edge_id: str = None):
        super().__init__(source, target, EdgeType.OPERAND, edge_id)


class Result(UseEdge):
    def __init__(self, source: Node, target: Node, edge_id: str = None):
        super().__init__(source, target, EdgeType.RESULT, edge_id)


class ComputeGraph:
    """计算图，支持嵌套结构"""

    def __init__(self, name: str = None, parent_graph: "ComputeGraph" = None):
        self.name = name or f"graph_{uuid.uuid4().hex[:8]}"
        self.parent_graph = parent_graph
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, UseEdge] = {}
        self.sub_graphs: List["ComputeGraph"] = []
        self.input_nodes: List[Value] = []
        self.output_nodes: List[Value] = []
        self.metadata: Dict[str, Any] = {}

    def add_node(self, node: Node) -> Node:
        """添加节点到计算图"""
        self.nodes[node.node_id] = node
        return node

    def add_edge(self, edge: UseEdge) -> UseEdge:
        """添加边到计算图"""
        self.edges[edge.edge_id] = edge
        # 更新节点的边列表
        edge.source.add_outgoing_edge(edge)
        edge.target.add_incoming_edge(edge)
        return edge

    def create_value_node(self, tensor_data: Any = None) -> Value:
        """创建值节点"""
        node = Value(tensor_data)
        self.add_node(node)
        return node

    def create_operation_node(
        self, operation_func, args: tuple = None, kwargs: dict = None
    ) -> Operation:
        """创建操作节点"""
        node = Operation.from_function(operation_func, args, kwargs)
        self.add_node(node)
        return node

    def connect_operand(self, source: Value, target: Operation) -> UseEdge:
        """连接操作数（输入）"""
        edge = UseEdge(source, target, EdgeType.OPERAND)
        self.add_edge(edge)
        return edge

    def connect_result(self, source: Operation, target: Value) -> UseEdge:
        """连接结果（输出）"""
        edge = UseEdge(source, target, EdgeType.RESULT)
        self.add_edge(edge)
        source.add_result(target)
        return edge

    def add_sub_graph(self, sub_graph: "ComputeGraph"):
        """添加子图"""
        sub_graph.parent_graph = self
        self.sub_graphs.append(sub_graph)

    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """根据ID获取节点"""
        return self.nodes.get(node_id)

    def get_nodes_by_type(self, node_type: NodeType) -> List[Node]:
        """根据类型获取节点"""
        return [node for node in self.nodes.values() if node.node_type == node_type]

    def get_operation_nodes(self) -> List[Operation]:
        """获取所有操作节点"""
        return [node for node in self.nodes.values() if isinstance(node, Operation)]

    def get_value_nodes(self) -> List[Value]:
        """获取所有值节点"""
        return [node for node in self.nodes.values() if isinstance(node, Value)]

    def __enter__(self):
        """上下文管理器入口"""
        global current_graph
        graph_stack.append(current_graph)
        current_graph = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        global current_graph
        current_graph = graph_stack.pop() if graph_stack else None

    def __repr__(self):
        return f"ComputeGraph(name={self.name}, nodes={len(self.nodes)}, edges={len(self.edges)})"

    @staticmethod
    def get_current_graph() -> "ComputeGraph":
        """静态方法：获取当前活跃的计算图"""
        global current_graph
        if current_graph is None:
            current_graph = ComputeGraph("default_graph")
        return current_graph


# 全局计算图实例
graph = ComputeGraph("global_graph")
current_graph = graph


def get_current_graph() -> ComputeGraph:
    """获取当前活跃的计算图"""
    global current_graph
    if current_graph is None:
        current_graph = ComputeGraph("default_graph")
    return current_graph


def append_to_graph(operation_func, args, kwargs, result):
    """将操作添加到当前计算图"""
    # 延迟导入以避免循环导入
    from .torch_inject import TensorLike

    if kwargs is None:
        kwargs = {}
    if args is None:
        args = tuple()

    current_graph = get_current_graph()

    # 创建操作节点
    op_node = current_graph.create_operation_node(operation_func, args, kwargs)

    # 处理输入参数，创建值节点并连接
    input_nodes = []
    for i, arg in enumerate(args):
        if isinstance(arg, TensorLike):
            # 为 TensorLike 对象创建值节点
            value_node = current_graph.add_node(Value(arg._tensor, f"arg_{i}"))
            current_graph.connect_operand(value_node, op_node)
            input_nodes.append(value_node)
        elif isinstance(arg, torch.Tensor):
            # 为原始 Tensor 创建值节点
            value_node = current_graph.add_node(Value(arg, f"arg_{i}"))
            current_graph.connect_operand(value_node, op_node)
            input_nodes.append(value_node)
        else:
            value_node = current_graph.add_node(Attr(arg, f"arg_{i}"))
            current_graph.connect_operand(value_node, op_node)
            input_nodes.append(value_node)

    # 处理关键字参数
    for key, value in kwargs.items():
        if isinstance(value, TensorLike):
            value_node = current_graph.add_node(Value(value._tensor, key))
            current_graph.connect_operand(value_node, op_node)
            input_nodes.append(value_node)
        elif isinstance(value, torch.Tensor):
            value_node = current_graph.add_node(Value(value, key))
            current_graph.connect_operand(value_node, op_node)
            input_nodes.append(value_node)
        else:
            value_node = current_graph.add_node(Attr(value, key))
            current_graph.connect_operand(value_node, op_node)
            input_nodes.append(value_node)

    # 处理结果，创建值节点并连接
    if result is not None:
        if isinstance(result, TensorLike):
            result_node = current_graph.add_node(Value(result._tensor, "result"))
            current_graph.connect_result(op_node, result_node)
        elif isinstance(result, torch.Tensor):
            result_node = current_graph.add_node(Value(result, "result"))
            current_graph.connect_result(op_node, result_node)
        elif isinstance(result, (int, float, bool, str)):
            result_node = current_graph.add_node(Attr(result, "result"))
            current_graph.connect_result(op_node, result_node)
        elif isinstance(result, (tuple, list)):
            # 处理多个返回值
            for item in result:
                if isinstance(item, TensorLike):
                    result_node = current_graph.add_node(Value(item._tensor, "result"))
                    current_graph.connect_result(op_node, result_node)
                elif isinstance(item, torch.Tensor):
                    result_node = current_graph.add_node(Value(item, "result"))
                    current_graph.connect_result(op_node, result_node)
        elif isinstance(result, np.ndarray):
            result_node = current_graph.add_node(ValueAttr(result, "result"))
            current_graph.connect_result(op_node, result_node)
        else:
            breakpoint()


def create_subgraph(module_name: str, suffix: str = "") -> ComputeGraph:
    """为 Module 创建子计算图"""
    global current_graph
    if suffix == "":
        sub_graph = ComputeGraph(f"{module_name}")
    else:
        sub_graph = ComputeGraph(f"{module_name}.{suffix}")
    if current_graph:
        current_graph.add_sub_graph(sub_graph)
    return sub_graph


def enter_subgraph(module_name: str, suffix: str = ""):
    """进入 Module forward 方法时调用"""
    global current_graph
    sub_graph = create_subgraph(module_name, suffix)
    graph_stack.append(current_graph)
    current_graph = sub_graph
    return sub_graph


def exit_subgraph():
    """退出 Module forward 方法时调用"""
    global current_graph
    if graph_stack:
        current_graph = graph_stack.pop()
    else:
        current_graph = graph


def get_graph_summary() -> Dict[str, Any]:
    """获取计算图摘要信息"""
    current_graph = get_current_graph()
    return {
        "graph_name": current_graph.name,
        "total_nodes": len(current_graph.nodes),
        "value_nodes": len(current_graph.get_value_nodes()),
        "operation_nodes": len(current_graph.get_operation_nodes()),
        "total_edges": len(current_graph.edges),
        "sub_graphs": len(current_graph.sub_graphs),
        "operations": [
            node.operation_name for node in current_graph.get_operation_nodes()
        ],
    }


def print_graph_summary():
    """打印计算图摘要"""
    current_graph = get_current_graph()
    print("=== 计算图摘要 ===")
    _print_graph_hierarchy(current_graph, indent_level=0)
    print("================")


def _print_graph_hierarchy(graph: ComputeGraph, indent_level: int = 0):
    """递归打印计算图层级结构"""
    indent = "  " * indent_level

    # 打印当前图信息
    print(f"{indent}{graph.name} {{")

    # 获取操作节点并按操作名称分组
    operation_nodes = graph.get_operation_nodes()
    if operation_nodes:
        for op_node in operation_nodes:
            op_indent = "  " * (indent_level + 1)
            print(
                f"{op_indent}{op_node.operation_name} (id={op_node.__class__.__name__})"
            )

            # 收集操作数节点
            operand_nodes = []
            for edge in op_node.incoming_edges:
                if edge.edge_type == EdgeType.OPERAND:
                    operand_nodes.append(edge.source)

            # 收集结果节点
            result_nodes = op_node.result_nodes

            # 打印节点信息（不单独展示 Operand 和 Result）
            if operand_nodes or result_nodes:
                node_indent = "  " * (indent_level + 1)
                if operand_nodes:
                    for node in operand_nodes:
                        print(f"{node_indent}-> {node}")

                if result_nodes:
                    for node in result_nodes:
                        print(f"{node_indent}<- {node}")

    # 递归打印子图
    if graph.sub_graphs:
        for sub_graph in graph.sub_graphs:
            _print_graph_hierarchy(sub_graph, indent_level + 1)

    print(f"{indent}}}")
