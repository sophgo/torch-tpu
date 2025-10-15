import json
import argparse
from pathlib import Path
import pandas as pd

from OpCounter import ConvBwdTiuTimeUs, ConvBwdDmaTimeUs, ConvTiuTimeUs, ConvDmaTimeUs, \
                      BNDmaTimeUs, BNTiuTimeUs, BNBackward_TiuTimeUs, BNBackward_DmaTimeUs
from utils import parse_bool_list, parse_int_list, parse_bool, parse_float

### aten::convolution_backward_overrideable
def parse_convolution_backward_dims(event_obj):
    args = event_obj.get("args", {})
    input_dims = args.get("Input Dims")
    concrete_inputs = args.get("Concrete Inputs")

    if not isinstance(input_dims, list) or len(input_dims) < 3:
        raise ValueError("Input Dims 缺失或结构不正确，期望至少包含 [grad_output, input, weight] 三项。")

    if not isinstance(concrete_inputs, list) or len(concrete_inputs) <= 9:
        # 若缺失或长度不足，默认开关全开
        gradinput_enable, gradweight_enable, gradbias_enable = True, True, True
    else:
        switches = parse_bool_list(concrete_inputs[9])
        if not switches or len(switches) < 3:
            gradinput_enable, gradweight_enable, gradbias_enable = True, True, True
        else:
            gradinput_enable, gradweight_enable, gradbias_enable = switches[:3]

    grad_out = input_dims[0]
    inp = input_dims[1]
    weight = input_dims[2]

    def ensure_4d(name, dims):
        if not isinstance(dims, list) or len(dims) != 4:
            raise ValueError(f"{name} 维度应为4D，实际为：{dims}")
        return dims

    grad_out = ensure_4d("grad_output", grad_out)
    inp = ensure_4d("input", inp)
    weight = ensure_4d("weight", weight)

    # 基础映射
    N_go, OC_go, OH, OW = grad_out
    N_in, IC_in, IH, IW = inp
    OC_w, IC_w, KH, KW = weight

    # 输出行（固定列）
    row = {
        # gradoutput
        "gradoutput_N": N_go,
        "gradoutput_OC": OC_go,
        "gradoutput_OH": OH,
        "gradoutput_OW": OW,
        # W
        "W_OC": OC_w,
        "W_IC": IC_w,
        "W_KH": KH,
        "W_KW": KW,
        # input
        "input_N": N_in,
        "input_IC": IC_in,
        "input_IH": IH,
        "input_IW": IW,
        # Bias
        "Bias_OC": OC_w
    }

    tiuus = ConvBwdTiuTimeUs(grad_out, inp, weight, gradinput_enable, gradweight_enable, gradbias_enable)
    dmaus = ConvBwdDmaTimeUs(grad_out, inp, weight, gradinput_enable, gradweight_enable, gradbias_enable)

    # 依据开关填充梯度列
    row.update({
        "gradinput_N": N_in if gradinput_enable else None,
        "gradinput_IC": IC_in if gradinput_enable else None,
        "gradinput_IH": IH if gradinput_enable else None,
        "gradinput_IW": IW if gradinput_enable else None,

        "gradweight_OC": OC_w if gradweight_enable else None,
        "gradweight_IC": IC_w if gradweight_enable else None,
        "gradweight_KH": KH if gradweight_enable else None,
        "gradweight_KW": KW if gradweight_enable else None,

        "gradbias_OC": (OC_w if gradbias_enable else None),
        "DMA(us)"    : dmaus,
        "TIU(us)"    : tiuus,
        "Alg(us)"    : max(dmaus, tiuus),
    })
    return row
def parse_convolution_backward(data):
    events = data.get("traceEvents", [])
    rows = []
    for ev in events:
        if ev.get("cat",None) != "cpu_op":
            continue
        if ev.get("name", "") != "aten::convolution_backward_overrideable":
            continue
        row = parse_convolution_backward_dims(ev)
        print(row)
        rows.append(row)
    return rows
def save_convbwd( data, out_path, sheet_name="convbwd"):
    rows = parse_convolution_backward(data)
    columns_order = [
        # gradoutput
        "gradoutput_N", "gradoutput_OC", "gradoutput_OH", "gradoutput_OW",
        # W
        "W_OC", "W_IC", "W_KH", "W_KW",
        # input
        "input_N", "input_IC", "input_IH", "input_IW",
        # Bias
        "Bias_OC",
        # extra gradient columns
        "gradinput_N", "gradinput_IC", "gradinput_IH", "gradinput_IW",
        "gradweight_OC", "gradweight_IC", "gradweight_KH", "gradweight_KW",
        "gradbias_OC", "DMA(us)", "TIU(us)", "Alg(us)" 
    ]
    df = pd.DataFrame(rows, columns=columns_order)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"已保存到：{out_path}-{sheet_name}")

### aten::convolution_overrideable
def parse_one_convolution_overrideable(event_obj):
    args = event_obj.get("args", {})
    # parser input dims
    input_dims = args.get("Input Dims")

    if not isinstance(input_dims, list) or len(input_dims) < 3:
        raise ValueError("Input Dims 缺失或结构不正确，期望至少包含 [input, weight, bias] 三项。")

    inp = input_dims[0]
    weight = input_dims[1]
    bias = input_dims[2]

    def ensure_4d(name, dims):
        if not isinstance(dims, list) or len(dims) != 4:
            raise ValueError(f"{name} 维度应为4D，实际为：{dims}")
        return dims

    inp = ensure_4d("input", inp)
    weight = ensure_4d("weight", weight)
    bias = bias if len(bias) == 1 else [0]
    # parser concrete inputs
    concrete_inputs = args.get("Concrete Inputs")
    stride          = parse_int_list(concrete_inputs[3])
    padding         = parse_int_list(concrete_inputs[4])
    dilation        = parse_int_list(concrete_inputs[5])
    transposed      = concrete_inputs[6]
    output_padding  = concrete_inputs[7]
    groups          = int(concrete_inputs[8])
    assert groups == 1, "only support groups == 1 now"

    N_in, IC_in, IH, IW = inp
    OC_w, IC_w, KH, KW  = weight
    OH = (IH + 2 * padding[0] - KH) // stride[0] + 1
    OW = (IW + 2 * padding[1] - KW) // stride[1] + 1
    # 输出行（固定列）
    row = {
        # input
        "input_N": N_in,
        "input_IC": IC_in,
        "input_IH": IH,
        "input_IW": IW,
        # W
        "W_OC": OC_w,
        "W_IC": IC_w,
        "W_KH": KH,
        "W_KW": KW,
        # Bias
        "Bias": bias[0],
        # output
        "output_N": N_in,
        "output_OC": OC_w,
        "output_OH": OH,
        "output_OW": OW,
    }

    tiuus = ConvTiuTimeUs(inp, weight, [N_in, OC_w, OH, OW], bias)
    dmaus = ConvDmaTimeUs(inp, weight, [N_in, OC_w, OH, OW], bias)

    # 依据开关填充梯度列
    row.update({
        "DMA(us)"    : dmaus,
        "TIU(us)"    : tiuus,
        "Alg(us)"    : max(dmaus, tiuus),
    })
    return row
def parse_convolution_overrideable(data):
    events = data.get("traceEvents", [])
    rows = []
    for ev in events:
        if ev.get("cat",None) != "cpu_op":
            continue
        if ev.get("name", "") != "aten::convolution_overrideable":
            continue
        row = parse_one_convolution_overrideable(ev)
        print(row)
        rows.append(row)
    return rows
def save_conv( data, out_path, sheet_name="conv"):
    rows = parse_convolution_overrideable(data)
    columns_order = [
        # input
        "input_N", "input_IC", "input_IH", "input_IW",
        # W
        "W_OC", "W_IC", "W_KH", "W_KW",
        # Bias
        "Bias",
        # output
        "output_N", "output_OC", "output_OH", "output_OW",
        "DMA(us)", "TIU(us)", "Alg(us)" 
    ]
    df = pd.DataFrame(rows, columns=columns_order)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"已保存到：{out_path}-{sheet_name}")

### aten::native_batch_norm
def parse_one_native_batch_norm(event_obj):
    args = event_obj.get("args", {})
    input_dims = args.get("Input Dims")

    if not isinstance(input_dims, list) or len(input_dims) < 5:
        raise ValueError("Input Dims 缺失或结构不正确，期望至少包含 [input, weight, bias, running_mean, running_var] 五项。")

    inp          = input_dims[0]  # [N, C, H, W]
    weight_gamma = input_dims[1]  # [C]
    bias_beta    = input_dims[2]  # [C]
    running_mean = input_dims[3]  # [C]
    running_var  = input_dims[4]  # [C]

    def ensure_4d(name, dims):
        if not isinstance(dims, list) or len(dims) != 4:
            raise ValueError(f"{name} 维度应为4D，实际为：{dims}")
        return dims

    def ensure_1d_c(name, dims, C_expected):
        if not isinstance(dims, list) or len(dims) != 1:
            raise ValueError(f"{name} 维度应为1D，实际为：{dims}")
        if dims[0] != C_expected:
            raise ValueError(f"{name} 的长度应与通道数一致，期望 {C_expected}，实际 {dims[0]}")
        return dims

    # 校验维度
    inp = ensure_4d("input", inp)
    N, C, H, W = inp
    weight_gamma = ensure_1d_c("weight(gamma)", weight_gamma, C)
    bias_beta    = ensure_1d_c("bias(beta)", bias_beta, C)
    running_mean = ensure_1d_c("running_mean", running_mean, C)
    running_var  = ensure_1d_c("running_var", running_var, C)

    # 解析 Concrete Inputs
    # Concrete Inputs: ["", "", "", "", "", "True", "0.029999999999999999", "0.001"]
    concrete_inputs = args.get("Concrete Inputs", [])
    if len(concrete_inputs) < 8:
        raise ValueError(f"Concrete Inputs 长度不足，实际为 {len(concrete_inputs)}：{concrete_inputs}")

    training_str = concrete_inputs[5]
    momentum_str = concrete_inputs[6]
    eps_str      = concrete_inputs[7]

    # 规范化解析
    training = str(training_str).strip().lower() == "true"
    try:
        momentum = float(str(momentum_str).strip()) if str(momentum_str).strip() != "" else 0.1
    except Exception:
        raise ValueError(f"momentum 解析失败：{momentum_str}")
    try:
        eps = float(str(eps_str).strip()) if str(eps_str).strip() != "" else 1e-5
    except Exception:
        raise ValueError(f"eps 解析失败：{eps_str}")

    # 输出形状与统计（BatchNorm 不改变形状）
    out_N, out_C, out_H, out_W = N, C, H, W

    # 这里不计算具体时间，留空或由外部函数计算
    row = {
        # input
        "input_N": N,
        "input_C": C,
        "input_H": H,
        "input_W": W,
        # params
        "gamma_C": weight_gamma[0],
        "beta_C": bias_beta[0],
        "running_mean_C": running_mean[0],
        "running_var_C": running_var[0],
        # hyper
        "training": training,
        "momentum": momentum,
        "eps": eps,
        # output
        "output_N": out_N,
        "output_C": out_C,
        "output_H": out_H,
        "output_W": out_W,
        # placeholder time/us
        "DMA(us)": None,
        "TIU(us)": None,
        "Alg(us)": None,
    }

    tiuus = BNTiuTimeUs(inp)
    dmaus = BNDmaTimeUs(inp)
    row.update({"DMA(us)": dmaus, "TIU(us)": tiuus, "Alg(us)": max(dmaus, tiuus)})

    return row
def parse_native_batch_norm(data):
    events = data.get("traceEvents", [])
    rows = []
    for ev in events:
        if ev.get("cat", None) != "cpu_op":
            continue
        if ev.get("name", "") != "aten::native_batch_norm":
            continue
        row = parse_one_native_batch_norm(ev)
        print(row)
        rows.append(row)
    return rows
def save_bn(data, out_path, sheet_name="batch_norm"):
    rows = parse_native_batch_norm(data)
    columns_order = [
        # input
        "input_N", "input_C", "input_H", "input_W",
        # params
        "gamma_C", "beta_C", "running_mean_C", "running_var_C",
        # hyper
        "training", "momentum", "eps",
        # output
        "output_N", "output_C", "output_H", "output_W",
        "DMA(us)", "TIU(us)", "Alg(us)"
    ]
    df = pd.DataFrame(rows, columns=columns_order)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"已保存到：{out_path}-{sheet_name}")

### aten::native_batch_norm_backward
def parse_one_native_batch_norm_backward(event_obj):
    args = event_obj.get("args", {})
    input_dims = args.get("Input Dims")

    # 期望至少包含 7 个张量维度项：[x, grad_out, weight, running_mean, running_var, saved_mean, saved_invstd, ...]
    if not isinstance(input_dims, list) or len(input_dims) < 7:
        raise ValueError("Input Dims 缺失或结构不正确，期望至少包含 [x, grad_out, weight, running_mean, running_var, saved_mean, saved_invstd] 七项。")

    x           = input_dims[0]  # [N, C, H, W]
    grad_out    = input_dims[1]  # [N, C, H, W]
    weight      = input_dims[2]  # [C]
    running_mean= input_dims[3]  # [C]
    running_var = input_dims[4]  # [C]
    saved_mean  = input_dims[5]  # [C]
    saved_invstd= input_dims[6]  # [C]

    def ensure_4d(name, dims):
        if not isinstance(dims, list) or len(dims) != 4:
            raise ValueError(f"{name} 维度应为4D，实际为：{dims}")
        return dims

    def ensure_1d_c(name, dims, C_expected):
        if not isinstance(dims, list) or len(dims) != 1:
            raise ValueError(f"{name} 维度应为1D，实际为：{dims}")
        if dims[0] != C_expected:
            raise ValueError(f"{name} 的长度应与通道数一致，期望 {C_expected}，实际 {dims[0]}")
        return dims

    # 校验维度一致性
    x = ensure_4d("x", x)
    grad_out = ensure_4d("grad_out", grad_out)
    Nx, Cx, Hx, Wx = x
    Ng, Cg, Hg, Wg = grad_out
    if (Nx, Cx, Hx, Wx) != (Ng, Cg, Hg, Wg):
        raise ValueError(f"x 与 grad_out 形状不匹配: x={x}, grad_out={grad_out}")

    weight       = ensure_1d_c("weight(gamma)", weight, Cx)
    running_mean = ensure_1d_c("running_mean", running_mean, Cx)
    running_var  = ensure_1d_c("running_var", running_var, Cx)
    saved_mean   = ensure_1d_c("saved_mean", saved_mean, Cx)
    saved_invstd = ensure_1d_c("saved_invstd", saved_invstd, Cx)

    # 解析 Concrete Inputs
    # 示例: ["", "", "", "", "", "", "", "True", "0.001", "[True, True, True]"]
    concrete_inputs = args.get("Concrete Inputs", [])
    if len(concrete_inputs) < 10:
        raise ValueError(f"Concrete Inputs 长度不足，实际为 {len(concrete_inputs)}：{concrete_inputs}")

    training = parse_bool(concrete_inputs[7])
    eps      = parse_float(concrete_inputs[8], default=1e-5)
    output_mask = parse_bool_list(concrete_inputs[9])
    # 规范化为长度3
    if len(output_mask) != 3:
        # 若日志有其它格式，尝试填充或截断
        if len(output_mask) < 3:
            output_mask = output_mask + [True] * (3 - len(output_mask))
        else:
            output_mask = output_mask[:3]
    need_grad_input, need_grad_weight, need_grad_bias = output_mask

    # 输出形状（梯度返回的形状）
    grad_input_shape = [Nx, Cx, Hx, Wx] if need_grad_input else None
    grad_weight_shape = [Cx] if need_grad_weight else None
    grad_bias_shape = [Cx] if need_grad_bias else None

    row = {
        # forward input
        "x_N": Nx, "x_C": Cx, "x_H": Hx, "x_W": Wx,
        # grad_out
        "go_N": Ng, "go_C": Cg, "go_H": Hg, "go_W": Wg,
        # params (length C)
        "weight_C": weight[0],
        "running_mean_C": running_mean[0],
        "running_var_C": running_var[0],
        "saved_mean_C": saved_mean[0],
        "saved_invstd_C": saved_invstd[0],
        # hyper
        "training": training,
        "eps": eps,
        # outputs requested
        "need_grad_input": need_grad_input,
        "need_grad_weight": need_grad_weight,
        "need_grad_bias": need_grad_bias,
        # output shapes
        "grad_input_N": grad_input_shape[0] if grad_input_shape else None,
        "grad_input_C": grad_input_shape[1] if grad_input_shape else None,
        "grad_input_H": grad_input_shape[2] if grad_input_shape else None,
        "grad_input_W": grad_input_shape[3] if grad_input_shape else None,
        "grad_weight_C": grad_weight_shape[0] if grad_weight_shape else None,
        "grad_bias_C": grad_bias_shape[0] if grad_bias_shape else None,
        # placeholder time/us
        "DMA(us)": None,
        "TIU(us)": None,
        "Alg(us)": None,
    }

    tiuus = BNBackward_TiuTimeUs(x)
    dmaus = BNBackward_DmaTimeUs(x)
    row.update({"DMA(us)": dmaus, "TIU(us)": tiuus, "Alg(us)": max(dmaus, tiuus)})

    return row
def parse_native_batch_norm_backward(data):
    events = data.get("traceEvents", [])
    rows = []
    for ev in events:
        if ev.get("cat", None) != "cpu_op":
            continue
        if ev.get("name", "") != "aten::native_batch_norm_backward":
            continue
        row = parse_one_native_batch_norm_backward(ev)
        print(row)
        rows.append(row)
    return rows
def save_bn_backward(data, out_path, sheet_name="batch_norm_backward"):
    rows = parse_native_batch_norm_backward(data)
    columns_order = [
        # x
        "x_N", "x_C", "x_H", "x_W",
        # grad_out
        "go_N", "go_C", "go_H", "go_W",
        # params
        "weight_C", "running_mean_C", "running_var_C", "saved_mean_C", "saved_invstd_C",
        # hyper
        "training", "eps",
        # output mask
        "need_grad_input", "need_grad_weight", "need_grad_bias",
        # output shapes
        "grad_input_N", "grad_input_C", "grad_input_H", "grad_input_W",
        "grad_weight_C", "grad_bias_C",
        # times
        "DMA(us)", "TIU(us)", "Alg(us)"
    ]
    df = pd.DataFrame(rows, columns=columns_order)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"已保存到：{out_path}-{sheet_name}")

### aten::mul


supported_list = [
    "aten::detach", "aten::result_type", "aten::set_", "aten::detach_",
    "aten::__and__", "aten::numpy_T",
    "aten::empty", "aten::empty_like", "aten::empty_strided", "aten::new_empty",
    "aten::view", "aten::reshape", "aten::as_strided", "aten::as_strided_", "aten::unsqueeze", "aten::squeeze", "aten::permute","aten::_unsafe_view",
    "aten::expand", "aten::lift_fresh", "aten::split_with_sizes",
    "aten::slice", "aten::narrow", "aten::split", "aten::chunk", "aten::unbind",

    "aten::convolution_backward", "aten::convolution_backward_overrideable",

    "aten::conv2d", "aten::convolution","aten::_convolution", "aten::convolution_overrideable",
    "aten::native_batch_norm_backward",
    "aten::batch_norm", "aten::_batch_norm_impl_index", "aten::native_batch_norm",

    "aten::max_pool2d_with_indices", "aten::max_pool2d", "aten::max_pool2d_with_indices_backward",
    "aten::upsample_nearest2d_backward", "aten::upsample_nearest2d", 

    "aten::mul","aten::mul_", "aten::div", "aten::div_", "aten::reciprocal", 
    "aten::add", "aten::add_", "aten::sub", "aten::sub_", "aten::bitwise_and", "aten::max",
    "aten::neg", "aten::neg_", "aten::nonzero",
    "aten::maximum", "aten::minimum", "aten::clamp",
    "aten::exp", "aten::exp_", "aten::log", "aten::log_", "aten::atan", "aten::pow",
    "aten::mean", "aten::sum", 
    "aten::lt", "aten::ge", "aten::gt", "aten::eq",
    "aten::remainder", "aten::rsub", "aten::clamp_min_", "aten::clamp_",
    "aten::sigmoid", "aten::silu_backward", "aten::silu_", "aten::silu", "aten::sigmoid_backward",
    
    "aten::to", "aten::_to_copy", "aten::copy_", "aten::contiguous", "aten::clone", "aten::_copy_from", "aten::repeat",

    "aten::zero_", "aten::fill_", "aten::full_like", "aten::zeros", "aten::ones_like", "aten::zeros_like", "aten::full",
    "aten::new_zeros", "aten::ones", "aten::arange", "aten::scalar_tensor",
    "aten::cat", "aten::stack",
    "aten::_local_scalar_dense", "aten::item",
    "aten::index_put_", "aten::_index_put_impl_", "aten::index_put", "aten::masked_fill_",
    "aten::index_select", "aten::index", "aten::select", "aten::select_backward",
    "aten::binary_cross_entropy_with_logits",
    "aten::where",
    ]

def parse_aten_op(data):
    events = data.get("traceEvents", [])
    i = 0
    for ev in events:
        name = ev.get("name", "")
        if "aten::" in name and name not in supported_list:
            print(ev.get("name", ""))
            i += 1
            print(i)


def main():
    parser = argparse.ArgumentParser(description="Parse convolution_backward JSON dims to Excel")
    parser.add_argument("-i", "--input", required=True, help="输入 JSON 文件路径（包含单条事件对象）")
    parser.add_argument("-o", "--output", default="yolov5s-train.xlsx", help="输出 Excel 文件路径")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    save_conv(data, out_path)
    save_convbwd(data, out_path)
    save_bn(data, out_path)
    save_bn_backward(data, out_path)
    parse_aten_op(data)

if __name__ == "__main__":
    main()