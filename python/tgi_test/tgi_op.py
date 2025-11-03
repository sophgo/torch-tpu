import itertools
import csv
import os
import sys
# os.environ['DISABLE_CACHE'] = '1'
# os.environ['CMODEL_GLOBAL_MEM_SIZE']='12000000000'
# os.environ['CMODEL_FAST_EXEC']='1'
model_params_mapping = {
    'qwen_7b': {
        'tp': [1, 4],
        'quantize': ['gptq'],
        'batches': [16],
        'cases': ['add', 'rmsnorm', 'mmqkv','attn', 'mlp', 'attn_fc'],
        'seqs': [4096]
    },
    'qwen_72b': {
        'tp': [2, 4, 8],
        'quantize': ['gptq'],
        'batches': [16],
        'cases': ['add', 'rmsnorm', 'mmqkv','attn', 'mlp', 'attn_fc'],
        'seqs': [4096]
    },
#    'llama2_7b': {
#        'tp': [1],
#        'quantize': ['gptq'],
#        'batches': [16],
#        'cases': ['add', 'rmsnorm', 'mmqkv','attn', 'mlp', 'attn_fc'],
#        'seqs': [4096]
#    },
#    'llama2_70b': {
#        'tp': [4,8],
#        'quantize': ['gptq'],
#        'batches': [16],
#        'cases': ['add', 'rmsnorm', 'mmqkv','attn', 'mlp', 'attn_fc'],
#        'seqs': [4096]
#    },
    'llama3_8b': {
        'tp': [1,4,8],
        'quantize': ['gptq'],
        'batches': [16],
        'cases': ['add', 'rmsnorm', 'mmqkv','attn', 'mlp', 'attn_fc'],
        'seqs': [4096]
    },
#    'llama3_70b': {
#        'tp': [2,4,8],
#        'quantize': ['gptq'],
#        'batches': [16],
#        'cases': ['add', 'rmsnorm', 'mmqkv','attn', 'mlp', 'attn_fc'],
#        'seqs': [4096]
#    }
}


# with open('failed_runs.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)

#     writer.writerow(['Model', 'Batch', 'TP', 'Quantize', 'Case', 'Seq', 'Error'])

for model, params in model_params_mapping.items():
    parameter_combinations = itertools.product(
        params['batches'],
        params['tp'],
        params['quantize'],
        params['cases'],
        params['seqs']
    )

    for batch, tp, quantize, case, seq in parameter_combinations:
        #decode
        command_dec = f'python -u tgi_test.py --model {model} --batch {batch} --tp {tp} --case {case} --seq {seq} --test'
        if quantize == 'gptq' and case in ['mmqkv', 'attn_fc', 'mlp']:
            command_dec += ' --w4a16'
        # if case in ['rmsnorm']:
        #     command_dec = f'DISABLE_CACHE=1 {command_dec}'
        print(f"Executing: {command_dec}")     
        return_code = os.system(command_dec)

        if return_code != 0:
            error_message = f"Command failed with return code: {return_code}"
            print(error_message)
            # writer.writerow([model, batch, tp, quantize, case, seq, error_message])
            sys.exit(255)
        else:
            print("Decode command executed successfully.")
        #prefill
        command_pre = f'python -u tgi_test.py --model {model} --batch {batch} --tp {tp} --case {case} --seq 6 --prefill --test'
        if case in ['mlp']:
            continue
        else:
            # if case in ['rmsnorm','mmqkv','add']:
            #     command_pre = f'DISABLE_CACHE=1 {command_pre}'
            print(f"Executing: {command_pre}")
            return_code = os.system(command_pre)
            if return_code != 0:
                error_message = f"Command failed with return code: {return_code}"
                print(error_message)
                # writer.writerow([model, batch, tp, quantize, case, seq, error_message])
                sys.exit(255)
            else:
                print("Prefill command executed successfully.")

print("All commands have been processed.")
