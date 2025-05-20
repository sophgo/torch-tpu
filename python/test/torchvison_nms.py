import torch
import torchvision
import torch_tpu
device = 'tpu'
def case1():
    boxes       = torch.randn((30000,4))
    scores      = torch.randn((30000))
    iou_thres   = 0.6
    out         = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
    
    boxes_tpu   = boxes.to(device)
    scores_tpu  = scores.to(device)
    out_tpu     = torch.ops.my_ops.nms(boxes_tpu, scores_tpu, iou_thres)

    diff = out - out_tpu.cpu()
    print(torch.max(abs(diff)))
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    case1()
