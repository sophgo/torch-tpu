import numpy as np

def top_compare(top_mlir_f : str, inp_f : str, out_ref_f : str):
    from tpu_mlir.python.tools.model_runner import mlir_inference
    
    input_ref  = np.load(inp_f, allow_pickle=True)
    input_ref  = {key: input_ref[key] for key in input_ref.files}
    output_ref = np.load(out_ref_f, allow_pickle=True)
    output_ref = {key: output_ref[key] for key in output_ref.files}
    
    top_outs = mlir_inference(inputs=input_ref, mlir_file=top_mlir_f, dump_all=False)
    for k in output_ref.keys():
        ref = output_ref[k]
        if k not in top_outs.keys():
            print(f"========================{k}")
            continue
        top = top_outs[k].reshape(ref.shape)
        abs_error = np.abs(top - ref)
        # 防止除以0，分母为0时相对误差设为np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_error = np.abs(top - ref) / np.where(np.abs(ref) > 1e-12, np.abs(ref), np.nan)
        
        max_abs_idx = np.unravel_index(np.nanargmax(abs_error), abs_error.shape)
        max_rel_idx = np.unravel_index(np.nanargmax(rel_error), rel_error.shape)
        if np.nanmax(abs_error) > 1:
            print(f"{k}, 绝对误差最大值: {np.nanmax(abs_error)},ref={ref[max_abs_idx]}, top={top[max_abs_idx]}")
            print(f"     相对误差最大值: {np.nanmax(rel_error)},ref={ref[max_rel_idx]}, top={top[max_rel_idx]}")

if __name__ == "__main__":
    out_ref_f = "DummyCompiler/fx_fwd_outputs.npz"
    inp_f     = "DummyCompiler/fx_fwd_inputs.npz"
    top_compare()