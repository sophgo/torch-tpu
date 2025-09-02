import json
import argparse
from typing import Any, Dict
import torch
import ast

excluded_op = [
    #has been full tested
    "reshape",
    "view",
    "fill_",
    "arange",
    "zero_",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "new_zeros",
    "full_like",
    "add_",
    "sum",
    "atan",
    "exp_",
    "log_",
    "t",
    "transpose",
    "expand",
    "flatten",
    "_unsafe_view",
    "contiguous",
    "copy_",
    "eq",
    "squeeze",
    "permute",
    "unsqueeze",
    "ne",
    "sub",
    "rsub",
    "div_",
    "add",
    "pow",
    "neg",
    "lt",
    "gt",
    "type_as",
    "unsqueeze_",
    "squeeze_",
    "numpy_T",
    "linalg_vector_norm",

    #parameter parse error
    "mkldnn_convolution",
    "resize_",
    'as_strided_',
    "to",
    "_batch_norm_impl_index",
    "as_strided",
    "set_",
    "_to_copy",
    "clone",

    ##input shape was lost in trace
    "cat",
    "stack",
    "_foreach_norm",
    "_foreach_mul_",
    "index",

    #not support cmp
    "empty",
    "empty_like",
    "_reshape_alias",
    "new_empty",
    "empty_strided",
    "result_type",
    "detach",
    "dropout",
    "unbind",
    "chunk",
    "split",
    "split_with_sizes",

    #tpudnn not support
    "nll_loss_nd",
    "nll_loss",
    "nll_loss_forward",
    "nll_loss_backward",
    "_log_softmax_backward_data",
    "bernoulli_",
    "__and__",##utest not support bool inputs
    "bitwise_and",

    ##dispatch_to_small op in tpu or cpu smaller impl
    "div",
    "batch_norm",
    "conv2d",
    "convolution",
    "max_pool2d",
    "_scaled_dot_product_flash_attention",
    "scaled_dot_product_attention",
    "_scaled_dot_product_flash_attention_backward",
    "unfold",
    "remainder",
    "binary_cross_entropy_with_logits",
    "silu_backward",

    ##will be tested in froward op with true indices
    "max_pool2d_with_indices_backward",
]

def str2list(v):
    vars = v.split(',')
    vars = [s.strip() for s in vars]
    while vars.count('') > 0:
        vars.remove('')
    return vars


def parse_concrete_value(s: str) -> Any:
    raw = s
    if s == "inf" or s =="-inf":
        return float(s)
    if s == "" or s == "nan":
        return None
    try:
        return ast.literal_eval(s)
    except Exception as e1:
        try:
            return json.loads(s)
        except Exception as e2:
            msg = (
                f"unable to parse: {raw!r}\n"
                f"ast.literal_eval error: {e1}\n"
                f"json.loads error: {e2}"
            )
            raise ValueError(msg)

def last_non_empty_index(lst):
    for i in range(len(lst) - 1, -1, -1):
        if lst[i]:
            return i
    return -1

def extract_aten_events(trace_json_path: str = "trace.json",
                        output_path: str = "test.json",
                        included_op: list = []) -> None:
    id = 0
    op_included = set()
    with open(trace_json_path, "r") as f:
        data = json.load(f)

    events = data.get("traceEvents", [])
    aten_events = dict()
    aten_op_names = set()
    included_op = ['_convolution' if x == 'convolution' else x for x in included_op]
    included_op = set(included_op)
    for ev in events:
        if ev.get("cat",None) != "cpu_op":
            continue

        name = ev.get("name", "")
        if (not name.startswith("aten::")):
            continue
        name = name.split("aten::")[1]
        if (getattr(torch.ops.aten, name, None) == None) or (name in excluded_op):
            continue
        if len(included_op)>0 and name not in included_op:
            continue
        arg = ev.get("args", {})
        input_dims = arg.get("Input Dims", [])
        # pass all cases without input tensor because in profile some op loss shape for weight_folding or other reason
        # and some ops without input tensor like zeros,ones,arange has been tested
        if sum(len(d) for d in input_dims) == 0:
            continue
        inputs_types = arg.get("Input type", [])
        concrete_inputs = arg.get("Concrete Inputs", [])

        assert(len(input_dims) == len(inputs_types) == len(concrete_inputs))
        def get(lst, i, default):
            return lst[i] if i < len(lst) else default

        inputs_shape = []
        inputs_dtype = []
        inputs_struct = []

        params_begin_index = last_non_empty_index(input_dims)

        for i in range(len(input_dims)):
            dim = get(input_dims, i, [])
            ty = get(inputs_types, i, "")
            concr_raw = get(concrete_inputs, i, "")
            concrete_value = parse_concrete_value(concr_raw)

            if i<= params_begin_index:
                if ty in ["ScalarList" , "Scalar", ""]:
                    inputs_shape.append(concrete_value)
                else:
                    inputs_shape.append(tuple(dim))
            else:
                inputs_struct.append(concrete_value)
            inputs_dtype.append(ty)


        if name in ["addmm", "contiguous", "clone", "gelu", "gelu_backward", "sub_"]:
            inputs_struct=[] # some case not need args (MemoryFormat? memory_format=None)

        if name in ["expand", "bernoulli_", "linalg_vector_norm"]:
            inputs_struct = inputs_struct[:-1]

        if name in ["add", "sub", "index"]:
            inputs_struct = []

        if name == "pad":
            if inputs_struct[1] is None:
                inputs_struct[1] = "constant"

        if name == "where":
            if len(inputs_shape) < 3:
                continue

        if name == "mean":
            if inputs_struct[-1] is None:
                inputs_struct.pop()

        if name in ["mul","mul_"]:
            if len(inputs_shape) < 2:
                continue

        if name == "index_put_":
            inputs_shape = [inputs_shape[0]]
            inputs_struct = inputs_struct[-1:]

        if name == "_index_put_impl_":
            inputs_shape = [inputs_shape[0]]
            inputs_struct = inputs_struct[-2:]

        md5 = hash((name+str(inputs_shape)+str(inputs_struct)))
        if md5 in op_included:
            continue
        op_included.add(md5)
        aten_op_names.add(name)
        aten_events[id]={"name":name, "shape":inputs_shape, "params":inputs_struct, "dtype":inputs_dtype}
        id+=1
    if len(included_op)>0:
        for op in aten_op_names:
            if op not in included_op:
                print(f"{op} not found, please be careful! \n maybe:\n\twas excluded\n\tspell is not correct\n\t{op} not in profile")
                return
    print("all ops: \n choose from below to test specific op")
    print("\n".join(sorted(aten_op_names)))
    with open(output_path, 'w') as f:
        json.dump(aten_events, f, indent=2)
    print("\ntest_ops.json generated!")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract operator parameters from origin JSON and save to JSON.")
    parser.add_argument("--profile",
                        default="profile.json",
                        help="path to json file")
    parser.add_argument("--output",
                        default="ops.json",
                        help="path to output json file")
    parser.add_argument("--included_op", type=str2list, default=list(),
                        help="If set, will extrave only given ops. i.e. convolution,relu,max_pool2d_with_indices")
    args = parser.parse_args()
    extract_aten_events(args.profile, args.output, args.included_op)


