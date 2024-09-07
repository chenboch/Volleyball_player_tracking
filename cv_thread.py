import typing
import cv2
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, Qt, QThread
from util import video_to_frame
import numpy as np
class CVThread(QThread):
    emit_signal = pyqtSignal(np.ndarray)
    _run_flag=True
    def __init__(self,image):
        super(CVThread, self).__init__()
        self.image=image
        self.is_finish=False
        self.points=[]

    def run(self):
        try:
            cv2.namedWindow("video",cv2.WINDOW_NORMAL)
            w ,h = self.image.shape[1] , self.image.shape[0]
            cv2.resizeWindow("video",w,h)
            cv2.imshow("video",self.image)
            cv2.setMouseCallback('video', self.image_event)
            cv2.waitKey(0)
            while True:
                if cv2.getWindowProperty('video', cv2.WND_PROP_VISIBLE) < 1:
                    if self.is_finish:
                        range_pts=[list(x) for x in self.points]
                        range_pts = np.array(range_pts,np.int32)
                        self.emit_signal.emit(range_pts)
                    else:
                        self.emit_signal.emit(np.array([]))
                    break
        except Exception as e:
            print(e)
            pass
    
    def image_event(self,event,x,y,flags,datas):
        th=5

        if event!=4 and event!=5:
            return
        
        image=self.image.copy()
        h,w,c=image.shape
        if event==5:    #delete point
            if len(self.points)>0:
                self.points.pop()
                self.is_finish=False
        elif  self.is_finish :
            # self._run_flag = False
            print("finish!!")
            return
        if event==4 :  #add point 
            #check border point 
            if abs(x-0)<th:
                x=0
            elif abs(x-w) <th:
                x=w
            if abs(y-0)<th:
                y=0
            elif abs(y-h) <th:
                y=h
            self.points.append((x,y))
            
        # if is last point make it turn to zero point
        if len(self.points)>3:
            dx=abs(self.points[-1][0]-self.points[0][0])
            dy=abs(self.points[-1][1]-self.points[0][1])
            if dx+dy<15:
                self.points[-1]=self.points[0]
                self.is_finish=True
                print("finish")
        #check if line closed
        color =(0,255,0)  if self.is_finish else (0,0,255)
        #draw point and line in image
        for i in range(len(self.points)):
            cv2.circle(image,self.points[i], 5,(255, 0, 0), 3)
            if i>0:
                cv2.line(image, self.points[i-1], self.points[i], color, 4)
        cv2.imshow("video",image)
        

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        print("stop set range thread")
        self.quit()
        self.wait()

class VideoToImagesThread(QThread):
    emit_signal = pyqtSignal([list,int,int])   
    _run_flag=True
    def __init__(self,video_path):
        super(VideoToImagesThread, self).__init__()
        self.video_path=video_path
    def run(self):
        # capture from web cam
        video_images, fps, count=video_to_frame(self.video_path)
        _run_flag=False
        self.emit_signal.emit(video_images, fps, count)
                
       
    
    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        # if self.cap!=None:
        #     self.cap.release()
        print("stop video to image thread")
    
    def isFinished(self):
        print(" finish thread")

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)   
    _run_flag=True
    cap=None
    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        
        while self._run_flag:
            ret, cv_img = cap.read()
            if not ret:
                break
            else:
                self.change_pixmap_signal.emit(cv_img)
                
        cap.release()
    
    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        # if self.cap!=None:
        #     self.cap.release()
        print("stop video thread")
    
    def isFinished(self):
        print("isFinished video thread")
       
    
    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        del self.image_list
        del self.video_size
        del self.video_name
        print("stop image to video thread")
    
    def isFinished(self):
        print(" finish thread")

class CVLineThread(QThread):
    emit_signal = pyqtSignal(np.ndarray)
    _run_flag=True
    def __init__(self,image):
        super(CVLineThread, self).__init__()
        self.image=image
        self.finished=False
        self.points=[]

    def run(self):
        try:
            cv2.namedWindow("video",cv2.WINDOW_NORMAL)
            w ,h = self.image.shape[1] , self.image.shape[0]
            cv2.resizeWindow("video",w,h)
            cv2.imshow("video",self.image)
            cv2.setMouseCallback('video', self.image_event)
            # cv2.setMouseCallback('video', self.image_event,[self.image,self.points,[self.finished]])
            cv2.waitKey(0)
            while True:
                if cv2.getWindowProperty('video', cv2.WND_PROP_VISIBLE) < 1:
                    if self.finished:
                        range_pts=[list(x) for x in self.points]
                        range_pts = np.array(range_pts,np.int32)
                        self.emit_signal.emit(range_pts)
                    else:
                        self.emit_signal.emit(np.array([]))
                    break
        except Exception as e:
            print(e)
            pass
    
    def image_event(self,event,x,y,flags,datas):
        th=5

        if event!=4 and event!=5:
            return
        
        image=self.image.copy()
        h,w,c=image.shape
        if event==5:    #delete point
            if len(self.points)>0:
                self.points.pop()
        if event==4 :  #add point 
            #check border point 
            if abs(x-0)<th:
                x=0
            elif abs(x-w) <th:
                x=w
            if abs(y-0)<th:
                y=0
            elif abs(y-h) <th:
                y=h
            self.points.append((x,y))
            
        # 2 points are present
        if len(self.points) == 2:
            self.finished = True
        # move the points if more points are added
        if len(self.points)>2:
            self.points.pop(0)

        #draw point and line in image
        for i in range(len(self.points)):
            cv2.circle(image,self.points[i], 5,(255, 0, 0), 3)
            if i>0:
                cv2.line(image, self.points[i-1], self.points[i], (0, 0, 255), 4)
        cv2.imshow("video",image)
        

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        print("stop set line thread")
        self.quit()
        self.wait()

class ShowTrackThread(QThread):
    timeout_signal = pyqtSignal([int])
    finished_signal = pyqtSignal()
    
    def __init__(self, total_frame, timeout_ms = 99) -> None:
        super(ShowTrackThread, self).__init__()
        self.timeout_ms = timeout_ms
        self.total_frame = total_frame

    def run(self):
        frame_num = 0
        while frame_num < self.total_frame:
            frame_num = frame_num + 1
            self.timeout_signal.emit(frame_num)
            self.msleep(self.timeout_ms)
            
        self.finished_signal.emit()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        print("stop tracking thread")
        self.quit()
        self.wait()

if __name__ == "__main__":
    test =VideoThread()
    test.start()
