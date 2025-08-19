import sys
import re
from typing import Dict, List, Optional, Sequence
import os


class Node(object):
    def __init__(self):
        self.depth = 1
        self.children: List = []
        self.text: Optional[str] = None

    def add_child(self, node):
        self.children.append(node)
        return True


class Table(Node):
    """
    .. doctest::

        >>> tbl = Table('My friends', ['Name', 'Major Project'])
        >>> tbl.add_item(('Ramki', 'Python'))
        >>> tbl.add_item(('Pradeepto', 'Kde'))
        >>> print(tbl)

        .. list-table:: My friends
            :header-rows: 1

            * -  Name
              -  Major Project
            * -  Ramki
              -  Python
    """

    def __init__(self, title: str = "", header: Optional[Sequence[str]] = None, width: Optional[Sequence[int]] = None):
        super().__init__()
        self.text = title
        self.header = list(header) if header else None
        self.width = list(width) if width else None

    def add_item(self, row: Sequence[str]):
        """
        Adds a new row to the table.

        :arg row: list of items in the table.
        """
        self.children.append([("`" + txt + "`" if txt else txt) for txt in row])

    def __repr__(self):
        def print_table(header: Sequence[str]) -> List[str]:
            items: List[str] = []
            for i, hdr in enumerate(header):
                if i == 0:
                    items.append(f"    * -  {hdr}")
                else:
                    items.append(f"      -  {hdr}")
            return items

        out = [f".. list-table:: {self.text}"]
        if self.width:
            out.append(f"    :widths: {str(self.width)[1:-1]}")
        if self.header:
            out.append("    :header-rows: 1\n")
            out.extend(print_table(self.header))
        else:
            out.append("    :header-rows: 0\n")

        for ch in self.children:
            out.extend(print_table(ch))

        return "\n".join(out)


def extract_my_ops_defs(text: str) -> set[str]:
    lib_pattern = re.compile(
        r'TORCH_LIBRARY\s*\(\s*[^,]+,\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)\s*\{(.*?)\}',
        re.S
    )
    m = lib_pattern.search(text)
    if m:
        block_text = m.group(2)
        handle_name = m.group(1)
    else:
        block_text = text
        handle_name = None

    def remove_comments(s: str) -> str:
        s = re.sub(r'//.*', '', s)
        s = re.sub(r'/\*.*?\*/', '', s, flags=re.S) 
        return s

    block_text = remove_comments(block_text)

    if handle_name:
        def_pattern = re.compile(
            rf'\b{re.escape(handle_name)}\s*\.\s*def\s*\(\s*"([^"]+)"',
            re.S
        )
    else:
        def_pattern = re.compile(
            r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*def\s*\(\s*"([^"]+)"',
            re.S
        )

    seen = set()
    if handle_name:
        for fn in def_pattern.findall(block_text):
            if fn not in seen:
                seen.add(fn)
    else:
        for _, fn in def_pattern.findall(block_text):
            if fn not in seen:
                seen.add(fn)

    return seen


class NativeOp:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    @staticmethod
    def op_sort_key(x: str) -> str:
        return "".join(ch for ch in x if ch != "_")

    @staticmethod
    def op_group_fun(x: str) -> str:
        try:
            _, op = x.split("::", 1)
        except ValueError:
            op = x
        ch = op[0] if op else "#"
        return ch.upper() if ch.isalpha() else "#"

    @staticmethod
    def _strip_comments(text: str) -> str:
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        text = re.sub(r"//.*", "", text)
        return text

    @staticmethod
    def _iter_cpp_files(root_dir: str):
        for dirpath, _, filenames in os.walk(root_dir):
            for fn in filenames:
                if fn.endswith(".cpp"):
                    yield os.path.join(dirpath, fn)

    @staticmethod
    def _normalize_basename(op_name: str) -> str:
        base = op_name.split(".", 1)[0]
        return base

    def _extract_ops_from_text(self, text: str) -> List[str]:
        results: List[str] = []

        lib_impl_pat = re.compile(
            r"TORCH_LIBRARY_IMPL\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)\s*\{(.*?)\}",
            re.S,
        )

        for ns, _backend, handle, body in lib_impl_pat.findall(text):
            impl_pat = re.compile(
                rf"\b{re.escape(handle)}\s*\.\s*impl\s*\(\s*\"([^\"]+)\"",
                re.S,
            )
            for op_name in impl_pat.findall(body):
                basename = self._normalize_basename(op_name)
                results.append(f"{ns}::{basename}")

        return results

    def _collect_ops(self) -> List[str]:
        all_ops: List[str] = []
        for path in self._iter_cpp_files(self.root_dir):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = f.read()
            except (OSError, UnicodeDecodeError):
                continue
            clean = self._strip_comments(raw)
            all_ops.extend(self._extract_ops_from_text(clean))

        unique_ops = sorted(set(all_ops), key=lambda s: (self.op_group_fun(s), self.op_sort_key(s)))
        return unique_ops

    def get_op_set(self) -> Dict[str, List[str]]:
        ops = self._collect_ops()
        op_set_sorted = sorted(ops, key=lambda s: (s[0].lower() if s else "", s.lower()))
        return op_set_sorted


class CustomOp:
    """
    Load custom ops defined in a given C++ file path with TORCH_LIBRARY(...).
    """
    def __init__(self, src_path: str):
        self.src_path = src_path
        self.black_set = {"dummy", "dummy_no_kernel_launch","enable_profile","disable_profile","dynlib_execute"}

    @staticmethod
    def _read_file(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def get_op_set(self) -> List[str]:
        content = self._read_file(self.src_path)
        my_op_set = extract_my_ops_defs(content)
        my_op_set = my_op_set - self.black_set

        my_op_set_sorted = sorted(my_op_set, key=lambda s: (s[0].lower() if s else "", s.lower()))
        return my_op_set_sorted

def main(argv: List[str]) -> int:
    if len(argv) < 4:
        print("Usage: generate_operation.py <native_ops_dir> <custom_ops_path> <output_path>")
        return 2

    native_ops_dir = argv[1]
    custom_ops_path = argv[2]
    output_path = argv[3]
    tables: List[Table] = []

    native_builder = NativeOp(root_dir=native_ops_dir)
    op_set = native_builder.get_op_set()

    try:
        custom_builder = CustomOp(src_path=custom_ops_path)
        my_op_set = custom_builder.get_op_set()
    except FileNotFoundError:
        my_op_set = []
        raise FileExistsError(f"can not find custom_ops file")

    if op_set:
        tbl = Table(title="NativeOp", header=None, width=[80])
        for op in op_set:
            tbl.add_item([op])
        tables.append(tbl)
    
    if my_op_set:
        tbl = Table(title="CustomOp", header=None, width=[80])
        for op in my_op_set:
            tbl.add_item([op])
        tables.append(tbl)

    # Write output
    parent_dir = os.path.dirname(os.path.abspath(output_path))
    if parent_dir and not os.path.isdir(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(str(x) for x in tables))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
