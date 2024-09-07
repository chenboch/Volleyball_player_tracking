import torch
import cv2
import argparse
import os
import numpy as np
import logging
# from torch2trt import torch2trt, TRTModule

from .timerr import Timer
from .visualizer import plot_tracking
from .pose_viz import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict, check_video_rotation, draw_ankle_point
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
    
    def init_tracker(self):
        self.tracker = byte_tracker.BYTETracker(self.args,frame_rate=50)

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
    
    def process_image(self,image,show_track=False,frame_id=0):
        timer = Timer()
        image = image.copy()
        pts,online_tlwhs,online_ids,online_scores = pose_points_yolo(self.detector, image, self.pose, self.tracker,self.args)
        pts_dict = {}
        if len(online_ids)>0:
            # timer_track.tic()
            timer.toc()
            if show_track:
                image = plot_tracking(
                        image, online_tlwhs, online_ids, frame_id=frame_id, fps=1/max(timer.average_time,0.00001)
                )
            for i, (bbox, pid) in enumerate(zip(online_tlwhs, online_ids)):
                pts_dict[pid] = [bbox]
            # person_ids = np.arange(len(pts), dtype=np.int32)
            # print(pts)
            if pts is not None :
                for i, (pt, pid) in enumerate(zip(pts, online_ids)):
                    pts_dict[pid].append(pt)
                    # image=draw_points_and_skeleton(image, pt, joints_dict()['coco']['skeleton'], person_index=pid,
                    #                                         points_color_palette='gist_rainbow', skeleton_color_palette='jet',points_palette_samples=10,confidence_threshold=0.3)
                    image = draw_ankle_point(image, pt)

            return pts_dict,image
        else:
            timer.toc()
            return None,image

    # def convert_to_trt(self,net, output_name, height, width):
    #     from ViTPose_trt import TRTModule_ViTPose
    #     self.pose = TRTModule_ViTPose(path='models/vitpose-b-multi-coco.engine',device='cuda:0')
    #     net.eval()
    #     img = torch.randn( 1,3, height, width).cuda()
    #     # img = img.cuda()
    #     print('Starting conversion')
    #     net_trt = torch2trt(net, [img],max_batch_size=10,fp16_mode=True)
    #     torch.save(net_trt.state_dict(), output_name)
    #     print('Conversion Successful!')

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

        #write to file
        # for pid in pts_list:
        #     test=np.array(pts_list[pid],dtype=object)
        #     np.save(os.path.join(dir_path,f"{pid}.npy"), test)
        #     print(pid,test.shape)  

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
    