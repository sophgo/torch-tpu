## 使用tpu训练yolov5s 
脚本 
```bash 
python3 train_fp16.py --img 640 --epoch 3 --data coco128.yaml --weights yolov5s.pt --workers 1 --batch-size 2 --device tpu --optimizer SGD
python3 train.py --img 640 --epoch 3 --data coco128.yaml --weights yolov5s.pt --workers 1 --batch-size 2 --device tpu --optimizer SGD
```