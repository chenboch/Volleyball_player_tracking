import torch
import cv2
import argparse
import os
import numpy as np
import logging
# from torch2trt import torch2trt, TRTModule
import csv
from .timerr import Timer
from .visualizer import plot_tracking
from .pose_viz import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict, check_video_rotation
from .pose_utils import pose_points_yolo
from .general_utils import polys_from_pose
from .logger_helper import CustomFormatter
from .tracker import byte_tracker
from .builder import build_model


dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)))
# import sys
# sys.path.insert(0, os.path.join(dir_path, 'configs'))
# print(os.path.join(dir_path,"configs","coco","ViTPose_base_coco_256x192.py"))
logger = logging.getLogger("Tracker !")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)
logger.propagate=False

class Vitpose_Model():
    def __init__(self,detector,config_name="ViTPose_base_coco_256x192",track_rate=50):
        # self.detector = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        self.detector = detector
        # torch.hub.load('WongKinYiu/yolov7', 'custom', dir_path+'//yolov7x.pt',  trust_repo=True)
        self.pose = build_model(os.path.join(dir_path,"configs","coco",f"{config_name}.py"),dir_path+'//models/vitpose-b.pth')
        self.pose.cuda().eval()
        self.args = self.add_parser().parse_args()
        self.init_tracker()
        # self.tracker = byte_tracker.BYTETracker(self.args,frame_rate=track_rate)
        
        #extra var
        self.history_dict = {}
        self.thres = 1.5 #thresold
    
    def init_tracker(self):
        self.tracker = byte_tracker.BYTETracker(self.args,frame_rate=50)

    #smooth joint position
    def smooth_datas(self, datas,pid):
        pre_datas = self.history_dict[pid]
        self.history_dict[pid] = datas
        current_datas = datas
        improve_data = datas
        for i in range(17):        
            dis = np.sqrt(np.square(current_datas[i][0] - pre_datas[i][0]) + np.square(
            current_datas[i][1] - pre_datas[i][1]))
            
            if dis < self.thres :
                improve_data[i] = pre_datas[i]
            else:    
                improve_data[i][0] = self.one_euro_f(current_datas[i][0],pre_datas[i][0])
                improve_data[i][1] = self.one_euro_f(current_datas[i][1],pre_datas[i][1])
        return improve_data
    
    def add_parser(self):
        parser = argparse.ArgumentParser("ByteTrack Demo!")
        parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")

        parser.add_argument('--trt',action='store_true')
        parser.add_argument('--trt_pose_only',action='store_true')
        # tracking args
        parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
        parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
        parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
        parser.add_argument(
            "--aspect_ratio_thresh", type=float, default=1.6,
            help="threshold for filtering out boxes of which aspect ratio are above the given value."
        )
        parser.add_argument('--min_box_area', type=float, default=15, help='filter out tiny boxes')
        parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
        return parser
    
    def id_switch(self, online_ids, revise_id):
        if not revise_id:
            return online_ids
        online_set = set(online_ids)
        for idx, (old_val, new_val) in enumerate(revise_id):
            if old_val in online_set and new_val in online_set:
                index_b = online_ids.index(old_val)
                index_a = online_ids.index(new_val)
                online_ids[index_b], online_ids[index_a] = new_val, old_val
            elif old_val in online_set and new_val not in online_set:
                index_o = online_ids.index(old_val)
                online_ids[index_o] = new_val
        return online_ids

    def process_image(self, image, show_track = False, frame_id = 0):
        timer = Timer()
        image = image.copy()
        image = image[540:,:1080]
        online_tlwhs, online_ids, online_scores =  pose_points_yolo(self.detector, image, self.pose)
        # online_ids = self.id_switch(online_ids,revise_id)
        pts_dict = {}
        if len(online_ids)>0:
            timer.toc()
            if show_track:
                image = plot_tracking(
                        image, online_tlwhs, online_ids, frame_id=frame_id, fps=1/max(timer.average_time,0.00001)
                )
            for i, (bbox, pid,score) in enumerate(zip(online_tlwhs, online_ids,online_scores)):
                tmp = np.array(bbox)
                tmp[1] += 540
                pts_dict[pid] = [tmp,score]
            
            return pts_dict,image
        
        else:
            timer.toc()
            return None,image

    def process_video_file(self,video,save=False):
        timer = Timer()
        video_base=os.path.basename(video)
        vid = cv2.VideoCapture(video)
        fw, fh = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        save_images=[]
        frame_id = 0
        timer = Timer()
        timer_track = Timer()
        timer_det = Timer()
        pts_list = {}
        while(1):
            timer.tic()
            ret, frame = vid.read()
            
            if ret:
                frame = cv2.resize(frame, (fw, fh))
                frame_orig = frame.copy()
                # timer_det.tic()
                pts, online_tlwhs, online_ids, online_scores = pose_points_yolo(self.detector, frame, self.pose, self.tracker, self.args)
                # timer_det.toc()
                # logger.info("FPS detector = %s",1./timer_det.average_time)
                
                # dets = res.xyxy[0]
                # dets = dets[dets[:,5] == 0.]
                for pid in pts_list:
                    pts_list[pid].append(np.zeros((17,3))) 
                if len(online_ids) > 0:
                    # timer_track.tic()
                    timer.toc()

                    online_im = plot_tracking(
                        frame_orig, online_tlwhs, online_ids, frame_id=frame_id, fps=1 / timer.average_time
                    )
                    # person_ids = np.arange(len(pts), dtype=np.int32)
                    if pts is not None: 
                        for i, (pt, pid) in enumerate(zip(pts, online_ids)):
                            if pid in pts_list:
                                pts_list[pid][-1] = pt
                            else:
                                pts_list[pid]=[np.zeros((17,3))]
                                for i in range(frame_id-1):
                                    pts_list[pid].append(np.zeros((17,3)))
                                pts_list[pid][-1] = pt
                            online_im = draw_points_and_skeleton(online_im, pt, joints_dict()['coco']['skeleton'], person_index=pid,
                                                                    points_color_palette='gist_rainbow', skeleton_color_palette='jet', points_palette_samples=10, confidence_threshold=0.3)

                else:
                    timer.toc()
                    online_im = frame_orig

                # cv2.imshow('frame',online_im)
                frame_id += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        if save: 
            dir_path=os.path.join("out_data",video_base.replace(".mp4",""))
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            writer = cv2.VideoWriter(os.path.join("out_video",video_base), fourcc, vid.get(cv2.CAP_PROP_FPS), (fw, fh))
            for image in save_images:
                 writer.write(image)
            writer.release()

        vid.release()
        return pts_list

if __name__ =="__main__":
    test=Vitpose_Model()
    img=cv2.imread(dir_path+"//test.webp")
    image,has_id=test.process_image(img)
    cv2.imshow('image',image)
    cv2.waitKey(0)
    
