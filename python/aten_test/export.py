import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.profiler import profile, ProfilerActivity, record_function
from parse_json import extract_aten_events, str2list

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(2))

        self.classifier = nn.Sequential(
            nn.Linear(400,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10),
            nn.Softmax())

    def forward(self,x):
        a1=self.feature_extractor(x)
        a1 = torch.flatten(a1,1)
        a2=self.classifier(a1)
        return a2

model_factory = {
    # cases  :  {models_method,  input_size, num_classes}
    "lenet": {'model':LeNet, 'input_size': (1, 32, 32), 'num_classes': 10},
    "resnet50": {'model': torchvision.models.resnet50, 'input_size': (3, 224, 224), 'num_classes': 1000},
    "vit_b_16": {'model': torchvision.models.vit_b_16, 'input_size': (3, 224, 224), 'num_classes': 1000},
    "swin_b": {'model': torchvision.models.swin_b, 'input_size': (3, 224, 224), 'num_classes': 1000},
    # notice: For inception_v3, you should manually call 'loss = loss_fn(out[0], target)'
    "inception_v3": {'model': torchvision.models.inception_v3, 'input_size': (3, 299, 299), 'num_classes': 1000},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract operator parameters from origin JSON and save to JSON.")
    parser.add_argument("--model", required = True, choices = list(model_factory.keys()))
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--output", default="test_ops.json", help="path to json file")
    parser.add_argument("--cases", type=str2list, default=list(),
                        help="If set, will extrave only given cases. i.e. convolution,convolution_backward,relu,max_pool2d_with_indices")
    args = parser.parse_args()

    batch  = args.batch_size
    model_parameter = model_factory.get(args.model, None)

    model = model_parameter['model']()
    inputs = torch.randn((batch, *model_parameter['input_size']), dtype=torch.float32)
    target = torch.randint(0, model_parameter['num_classes'], (batch,), dtype=torch.int64)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        out = model(inputs)
        if args.model == "inception_v3":
            out = out[0]
        loss = loss_fn(out, target)
        loss.backward()

    profile_name = f"profile_{args.model}_b{batch}.json"
    prof.export_chrome_trace(profile_name)
    print(f"{profile_name} generated!")
    extract_aten_events(profile_name, args.output , args.cases)

