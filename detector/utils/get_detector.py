import torch
import os
dir_name=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
from detector import YOLOV7Model
def get_detector_model(model_name):
    model_path=os.path.join(dir_name,'yolo',model_name+".pt")
    if "v5" in model_name:
        yolov5=torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        return yolov5
    if "v7" in model_name:
        model=YOLOV7Model()
        model.init_model()
        return model
    return torch.hub.load('WongKinYiu/yolov7', 'custom',os.path.join(dir_name,'yolo',"yolov7.pt"))