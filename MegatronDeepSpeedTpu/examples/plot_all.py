import numpy as np
import nnmoduletools as utils
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description='Plot all')
    parser.add_argument('--rank', type=int, default=0, help='rank')
    parser.add_argument('--eg', type=str, default="bert", choices=["bert", "gpt", "llama"], help='eg')
    parser.add_argument("--steps", type=int, default=[1], nargs='+', help="step")
    parser.add_argument('-t', '--tolerance', type=str, default="0.99,0.90", help='Tolerance for the comparison, cosine and euclid. Default is 0.99,0.90')
    parser.add_argument('-a', "--abs_tol", type=float, default=1e-8, help="The absolute tolerance for plot. Default is 1e-8")
    parser.add_argument('-r', "--rel_tol", type=float, default=1e-3, help="The relative tolerance for plot. Default is 1e-3")
    parser.add_argument('-v', '--verbose', type=int, default=3, help="Set the verbose level. 0: quiet, 1: normal, 2: dump failed, 3: dump and plot failed, 4: dump and plot all.  Default is 1.")
    parser.add_argument('-s', '--summary', action='store_true', help='If set, only print the summary of the comparison')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    rank = args.rank
    eg = args.eg
    verbose = f"{args.verbose}"
    tolerance = list(map(float, args.tolerance.split(',')))


    latest = utils.module_debugger.LogReader(eg, rank=rank)
    print(f"{latest.tpu_dir = }")
    print(f"{latest.cuda_dir = }")
    
    # Check write permission
    if not os.path.exists(str(latest.tpu_dir)) or not os.path.exists(str(latest.cuda_dir)):
        print(f"Please make sure the directories {latest.tpu_dir} and {latest.cuda_dir} exist.")
        exit(1)
    
    if not os.access(str(latest.tpu_dir), os.W_OK) or not os.access(str(latest.cuda_dir), os.W_OK):
        print(f"Please make sure you have write permission for {latest.tpu_dir} and {latest.cuda_dir}")
        exit(1)
    
    for step in args.steps:
        print(f"Step {step}")
        print("Output")
        output_dir = latest.tpu_dir / f"compare_step_{step}"
        fn = f"results/rank_{rank}_output_{step}.npz"
        results_comparer = utils.NPZComparer(latest.tpu_dir / fn, latest.cuda_dir / fn)
        results_comparer.report(tolerance=tolerance,
                                abs_tol=args.abs_tol,
                                rel_tol=args.rel_tol, verbose=args.verbose, summary=args.summary,
                                output_dir=output_dir,
                                title=f"Output Step {step}", output_fn=f"compare_output_step_{step}.md")
        
        print("Input")
        inputs_fn = f"results/rank_{rank}_input_{step}.npz"
        inputs_comparer = utils.NPZComparer(latest.tpu_dir / inputs_fn, latest.cuda_dir / inputs_fn)
        inputs_comparer.report(tolerance=tolerance,
                               abs_tol=args.abs_tol,
                               rel_tol=args.rel_tol, verbose=args.verbose, summary=args.summary,
                               output_dir=output_dir,
                               title=f"Input Step {step}", output_fn=f"compare_input_step_{step}.md")
            
        # delta params
        print("Delta Params")
        params_before_fn = f"params/rank_{rank}_params_{step - 1}.npz"
        params_after_fn = f"params/rank_{rank}_params_{step}.npz"
        tpu_param_before = np.load(latest.tpu_dir / params_before_fn)
        tpu_param_after = np.load(latest.tpu_dir / params_after_fn)
        cuda_param_before = np.load(latest.cuda_dir / params_before_fn)
        cuda_param_after = np.load(latest.cuda_dir / params_after_fn)
        delta_param_tpu = {param: tpu_param_after[param] - tpu_param_before[param] for param in tpu_param_before}
        delta_param_cuda = {param: cuda_param_after[param] - cuda_param_before[param] for param in cuda_param_before}
        params_comparer = utils.NPZComparer(delta_param_tpu, delta_param_cuda)
        params_comparer.report(tolerance=tolerance,
                               abs_tol=args.abs_tol,
                               rel_tol=args.rel_tol, verbose=args.verbose, summary=args.summary,
                               output_dir=output_dir,
                               title=f"Delta Params Step {step}", output_fn=f"compare_params_step_{step}.md")
        
        # grads
        print("Grads")
        grads_fn = f"debug/rank_{rank}_bit16_groups_grads.npz"
        if step > 1:
            grads_fn = f"debug/rank_{rank}_bit16_groups_grads_{step - 1}.npz" 
        grads_comparer = utils.NPZComparer(latest.tpu_dir / grads_fn, latest.cuda_dir / grads_fn)
        grads_comparer.report(tolerance=tolerance,
                              abs_tol=args.abs_tol,
                              rel_tol=args.rel_tol, verbose=args.verbose, summary=args.summary,
                              output_dir=output_dir,
                              title=f"Grads Step {step}", output_fn=f"compare_grads_step_{step}.md")