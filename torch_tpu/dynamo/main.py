import torch
import os
import copy
import argparse
from tpu_mlir_jit import aot_backend
import tpu_mlir_jit as tpu_mlir_jit
import pdb
import torch_tpu
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip", default="bm1684x", choices=['bm1684x', 'bm1690'],
                        help="chip name")
    parser.add_argument("--debug", default="",
                        help="debug")
    parser.add_argument("--cmp", action='store_true',
                        help="enable cmp")
    parser.add_argument("--skip_tpu_mlir", action='store_true',
                        help="skip_tpu_mlir")
    parser.add_argument("--model", default="",help="model name")
    parser.add_argument("--only_test_bwd", action='store_true',help="only_test_bwd")
  
    args = parser.parse_args()
    tpu_mlir_jit.args = args
    tpu_dev = "tpu:0"
    device = torch.device(tpu_dev)
    if args.model == "resnet50":
        input = torch.randn((1, 3, 224, 224))
        import torchvision.models as models
        mod = models.resnet50()
        net_d = copy.deepcopy(mod)
        net_d.to(device)
        net_d.train()
        input_d = input.to(device)
        optimizer = torch.optim.SGD(net_d.parameters(), lr=0.01)
        optimizer.zero_grad()
        model_opt = torch.compile(net_d, backend=aot_backend)
        loss_d = model_opt(input_d)
        loss_d[0,0].backward()
        optimizer.step()
    elif args.model == "bert_large":
        from transformers import BertTokenizer, BertModel
        mod = BertModel.from_pretrained('bert-large-cased')
        tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        example_text = 'hello world!'
        bert_input = tokenizer(example_text,padding='max_length',
                            max_length = 10,
                            truncation=True,
                            return_tensors="pt")
        mask = bert_input['attention_mask'].to(device)
        input_id = bert_input['input_ids'].to(device)
        net_d = copy.deepcopy(mod)
        net_d.to(device)
        net_d.train()
        optimizer = torch.optim.SGD(net_d.parameters(), lr=0.01)
        optimizer.zero_grad()
        model_opt = torch.compile(net_d, backend=aot_backend)
        out = model_opt(input_id,mask)
        loss_d = out[0][:, 0, :]  # [batch, 768]
        loss_d[0,0].backward()
        optimizer.step()