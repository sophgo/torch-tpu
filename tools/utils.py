import ast

def parse_bool_list(s):
    """
    将字符串形式的布尔/数字列表解析为纯布尔列表。
    例如: "[True, true, 1, 0, False]" -> [True, True, True, False, False]
    """
    if not isinstance(s, str):
        return None
    try:
        # 先尝试用 ast.literal_eval 解析为 Python 列表
        val = ast.literal_eval(s)
    except Exception:
        # 兼容小写 true/false：替换后再解析
        s_norm = s.replace("true", "True").replace("false", "False")
        try:
            val = ast.literal_eval(s_norm)
        except Exception:
            return None
    if not isinstance(val, (list, tuple)):
        return None
    def to_bool(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return bool(x)
        if isinstance(x, str):
            xs = x.strip().lower()
            if xs in ("true", "t", "yes", "y", "1"):
                return True
            if xs in ("false", "f", "no", "n", "0"):
                return False
        # 其它类型或无法识别，按 True/False 默认False
        return False
    return [to_bool(x) for x in val]

def parse_int_list(s):
    """
    将字符串形式的整型数字列表解析为整型列表。
    例如: "[3, 2, 1, 0]" -> [3, 2, 1, 0]
    """
    if not isinstance(s, str):
        return None
    try:
        # 先尝试用 ast.literal_eval 解析为 Python 列表
        val = ast.literal_eval(s)
    except Exception:
        return None
    if not isinstance(val, (list, tuple)):
        return None
    def toint(x):
        if isinstance(x, int):
            return x
        if isinstance(x, str):
            xs = x.strip().lower()
            return int(xs)
        assert 0
        return 0
    return [toint(x) for x in val]

def parse_bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("true", "1"):
        return True
    if s in ("false", "0"):
        return False
    raise ValueError(f"布尔解析失败: {x}")

def parse_float(x, default=None):
    s = str(x).strip()
    if s == "" and default is not None:
        return default
    try:
        return float(s)
    except Exception:
        raise ValueError(f"浮点解析失败: {x}")