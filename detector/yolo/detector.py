import argparse
import time
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
sys.path.insert(0,os.path.join(os.path.dirname(__file__)))

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages,LoadImageArray
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer



def set_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r"Db/checkpoints/yolov7-d6.pt", help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms',default=True, action='store_true',help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    parser.add_argument('--hide-labels-name', default=False, action='store_true', help='hide labels')

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=300, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=True, action='store_true',
                        help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=True, action="store_true", help="with ReID module.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"Db/checkpoints/mot17_sbs_S50.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')

    opt = parser.parse_args()

    opt.jde = False
    opt.ablation = False
    return opt

class YOLOV7Model(object):
    def __init__(self,opt = set_opts()):
        self.opt=opt
        self.weights, self.view_img, self.save_txt, self.imgsz, self.trace =\
            self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size, self.opt.trace
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        ## not conifg var
        self.dataset=None
    


    def init_model(self):
        #create traker
        self.tracker = BoTSORT(self.opt, frame_rate=30.0)
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int( self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s= self.stride)  # check img_size
        if self.trace:
            self.model = TracedModel(self.model, self.device, self.imgsz)

        if self.half:
            self.model.half()  # to FP16

        # Second-stage classifier
        # classify = False
        # if classify:
        #     modelc = load_classifier(name='resnet101', n=2)  # initialize
        #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        
        return self.model

    def set_dataloaders(self,images):
        cudnn.benchmark = True
        if isinstance(images[0],str):
            self.dataset = LoadImages(images, img_size=self.imgsz, stride=self.stride)
        else:
            self.dataset = LoadImageArray(images, img_size=self.imgsz, stride=self.stride)
        
    
    def run_inference(self):
        #speed up & reduce memory usage
        with torch.no_grad():
            # run image in self.dataset
            for path, img, im0s, vid_cap in self.dataset:
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # # Warmup
                # if self.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                #     old_img_b = img.shape[0]
                #     old_img_h = img.shape[2]
                #     old_img_w = img.shape[3]
                #     for i in range(3):
                #         self.model(img, augment=self.augment)[0]

                t1 = time_synchronized()
                pred = self.model(img, augment=self.opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
                t2 = time_synchronized()

                # Apply Classifier
                # if classify:
                #     pred = apply_classifier(pred, modelc, img, im0s)
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0, frame = path, '', im0s, getattr(self.dataset, 'frame', 0)
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):

                        # 調整框的尺寸從 img_size 到 im0 大小
                        boxes = scale_coords(img.shape[2:], det[:, :4], im0.shape)
                        boxes = boxes.cpu().numpy()
                        detections = det.cpu().numpy()
                        valid_indices = detections[:, 5] == 0.0  # 選擇類別為人的偵測
                        detections = detections[valid_indices]
                        detections[:, :4] = boxes[valid_indices]
                
                        #track id
                        online_targets = self.tracker.update(detections, im0)
            
        if self.view_img:
            cv2.imshow(str(p), im0)
            cv2.waitKey(0)  # 1 millisecond
        return online_targets
    


if __name__ =="__main__":
    test=YOLOV7Model()
    test.init_model()
    img=cv2.imread("test.jpg")
    test.set_dataloaders([img])
    test.run_inference()