import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap, QIcon
from sports_ui import Ui_MainWindow
import os
import cv2
from timer import Timer
import numpy as np
from Pose_2D import Vitpose_Model
from cv_thread import CVThread,VideoToImagesThread, CVLineThread, ShowTrackThread
from pose_viz import draw_points_and_skeleton,joints_dict
from detector.utils.get_detector import get_detector_model
from util import DataType
import csv
import pandas as pd

from Pose_2D.vit_dir.pose_utils import get_pose_points
from one_euro_filter import OneEuroFilter
from display_window import DisplayWindow
from one_euro_filter_two_input import OneEuroFilter_Two_Input
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "Tactical Analysis - Integration V6B2(Player + Ball)"))
        
        self.init_model()
        self.init_var()
        self.bind_ui()
        self.setup_icon()
               
    def init_model(self):
        self.detector=get_detector_model("v7")
        self.pose2d=Vitpose_Model(self.detector)
        self.timer=Timer()

    def close_thread(self,thread):
        thread.stop()
        thread = None
        self.is_threading=False

    def init_var(self):
        self.db_path = "./../../Db/"
        self.is_play=False
        self.processed_images=-1
        self.fps=30
        self.video_images=[]
        self.video_path = ""
        self.is_threading=False
        self.video_scene = QGraphicsScene()
        self.video_scene.clear()
        self.ui.bbox_check.setChecked(False)
        self.pose2d.init_tracker()
        
        # True when saving video
        self.saving_video = False
        # load parameter of middle line
        self.line_parameter = list((float(line[0]), float(line[1])) for line in csv.reader(open(self.db_path + 'data/line_parameter.csv'))) if os.path.exists(self.db_path + "data/line_parameter.csv") else None # y = ax + b, (a, b)
        # scene for virtual court
        self.court_scene = QGraphicsScene()
        self.court_scene.clear()
        # load perspective matrix
        self.perspective_matrix = np.loadtxt(self.db_path + "data/matrix.csv",
                 delimiter=",", dtype=np.float32) if os.path.exists(self.db_path + "data/matrix.csv") else None
        # store revised id
        self.revise_id=[]
        self.total_images = 0
        # draw virtual court
        virtual_court_points = self.draw_court()
        self.ui.virtual_resolution_label.setText("(0, 0) - {}x{}".format(self.ui.Virtual_Court.width(), self.ui.Virtual_Court.height()))
        # get border in virtual court
        self.min_virtual_court_short_side = virtual_court_points[0][0]
        self.max_virtual_court_short_side = virtual_court_points[2][0]

        self.min_virtual_court_long_side = virtual_court_points[0][1]
        self.max_virtual_court_long_side = virtual_court_points[2][1]
        # store the accepted ID and rejected ID respectively in a list
        self.allowed_id_list = []
        self.rejected_id_list = []
        # store all points in this dataframe
        self.player_points_df = pd.DataFrame(columns = ['frame_number', 'object_id', 'remapped_id', 'Bbox_x', 'Bbox_y', 'Bbox_w', 'Bbox_h', 'virtual_x', 'virtual_y'])
        self.show_tracking = False
        # add remapped player number
        self.id_mapping = {}
        self.id_map = {i : [[],False] for i in range(1, 13)}
        
        # display window for DEMO
        self.display_window = DisplayWindow(self)
        # serving and reception data
        self.serving_reception_data = None
        #smooth for two input 
        self.one_euro_f = OneEuroFilter_Two_Input()
        #tracklet bufffer 
        self.tracklet = []
        self.tracking_line = []

        self.blue_team_switched_id = {}
        self.green_team_switched_id = {}

    def setup_icon(self):
        self.play_icon = QIcon(QPixmap(self.db_path + "/icons/play.png"))
        self.ui.play_btn.setIcon(self.play_icon)
        self.pause_icon = QIcon(QPixmap(self.db_path + "/icons/pause.png"))
        self.ui.frame_left_btn.setIcon(QIcon(QPixmap(self.db_path + "/icons/rewind.png")))
        self.ui.frame_right_btn.setIcon(QIcon(QPixmap(self.db_path + "/icons/forward.png")))
        # self.ui.video_btn.setIcon(QIcon(QPixmap(self.db_path + "/icons/open.png")))
        # self.ui.store_btn.setIcon(QIcon(QPixmap(self.db_path + "/icons/save.png")))
        
        self.ui.play_btn.setText("")
        self.ui.frame_left_btn.setText("")
        self.ui.frame_right_btn.setText("")
        # self.ui.video_btn.setText("")
        # self.ui.store_btn.setText("")

    def bind_ui(self):
        self.ui.video_btn.clicked.connect(
            lambda: self.load_video(self.ui.video_label, self.db_path + "/videos/"))
        self.ui.frame_slider.valueChanged.connect(self.analyze_frame)
        self.ui.play_btn.clicked.connect(self.play_btn_clicked)
        # draw roi button
        self.ui.bbox_check.clicked.connect(self.change_image_canvas) 
        self.ui.store_btn.clicked.connect(self.save_video)
        self.ui.DrawLine_btn.clicked.connect(self.start_draw_line_thread)
        # self.ui.line_check.clicked.connect(self.show_line) # TODO: decide how to deal with boundary case, e.g. attempt to draw the line when playing
        self.ui.Perspective_btn.clicked.connect(self.draw_perspective_points_start)
        self.ui.frame_left_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() - 1)
        )
        self.ui.frame_right_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() + 1)
        )
        self.ui.revise_btn.clicked.connect(self.revise_btn_clicked)
        self.ui.player_track_btn.clicked.connect(self.start_tracking)
        self.ui.start_btn.clicked.connect(self.play_btn_clicked)
        self.ui.load_csv_btn.clicked.connect(self.load_csv)
        self.ui.save_csv_btn.clicked.connect(self.save_csv)
        self.ui.draw_figure_btn.clicked.connect(self.draw_figure)
        self.ui.virtual_point_figure_checkbox.clicked.connect(self.swap_figure_y_limit)

        self.ui.remove_btn.clicked.connect(self.is_not_player)
        #analysis groupbox
        self.ui.accuracy_label.setVisible(False)
        self.ui.display_window_btn.clicked.connect(self.show_display_window)

    def change_image_canvas(self):
        if not self.is_play and self.total_images > 0:
            frame_num = self.ui.frame_slider.value()
            self.show_image(self.show_BBox(frame_num), self.video_scene, self.ui.Video_View)
        pass

    def show_image(self,image: np.ndarray, scene: QGraphicsScene, GraphicsView: QGraphicsView):
        
        scene.clear()
        image=image.copy()
        
        def resize_image(image, GraphicsView: QGraphicsView):
            image_w ,image_h = image.shape[1] ,image.shape[0]
            if image_w > image_h :
                scale = image_h/image_w
                w = GraphicsView.width()-5
                h = w*scale
            else :
                scale = image_w/image_h
                h = GraphicsView.height()-5
                w = h*scale
            image = cv2.resize(image, (int(w),int(h)), interpolation = cv2.INTER_AREA)
            return image

        image = resize_image(image, GraphicsView)
        w,h = image.shape[1],image.shape[0]
        bytesPerline = 3 * w
        qImg = QImage(image, w, h, bytesPerline, QImage.Format_RGB888).rgbSwapped()   
        scene.addPixmap(QPixmap.fromImage(qImg))
        GraphicsView.setScene(scene) 

    def load_data(self, label_item, dir_path="", value_filter=None, mode=DataType.DEFAULT):
        data_path = None
        if mode == DataType.FOLDER:
            data_path = QFileDialog.getExistingDirectory(
                self, mode.value['tips'], dir_path)
        else:
            name_filter = mode.value['filter'] if value_filter == None else value_filter
            data_path, _ = QFileDialog.getOpenFileName(
                None, mode.value['tips'], dir_path, name_filter)
        if label_item == None:
            return data_path
        # check exist
        if data_path:
            label_item.setText(os.path.basename(data_path))
            label_item.setToolTip(data_path)
        else:
            label_item.setText(mode.value['tips'])
            label_item.setToolTip("")
        return data_path      
         
    def load_video(self, label_item, path):
        if self.is_play:
            QMessageBox.warning(self, "讀取影片失敗", "請先停止播放影片!")
            return
        self.init_var()
        self.video_path = self.load_data(
                label_item, path, None, DataType.VIDEO)       
        # no video found
        if self.video_path == "":
            return
        label_item.setText("讀取影片中...")
        #run image thread
        self.v_t=VideoToImagesThread(self.video_path)
        self.v_t.emit_signal.connect(self.video_to_frame)
        self.v_t.start()

    def video_to_frame(self,video_images, fps, count):
        self.total_images = count
        self.ui.frame_slider.setMinimum(0)
        self.ui.frame_slider.setMaximum(count - 1)
        self.ui.frame_slider.setValue(0)
        self.ui.frame_label.setText(f'0/{count-1}')
        # show the first image of the video
        self.video_images=video_images
        self.show_image(self.video_images[0], self.video_scene, self.ui.Video_View)
        self.ui.video_label.setText(os.path.basename(self.video_path))
        height, width, _ = self.video_images[0].shape
        self.ui.image_resolution_label.setText("(0, 0) - {}x{}".format(width, height))
        self.close_thread(self.v_t)
        self.fps = fps

        self.init_figure(count)
        
    def analyze_frame(self, show_fps=True):
        frame_num = self.ui.frame_slider.value()

        self.ui.start_frame_num_spinbox.setValue(frame_num)
        self.ui.end_frame_num_spinbox.setValue(self.processed_images)
        #print(self.revise_id)
        self.ui.frame_label.setText(
            f'{frame_num}/{self.total_images - 1}')

        # no image to analyze
        if self.total_images <= 0:
            return
        
        # in tracking mode
        if self.show_tracking:
            return
        
        if frame_num > self.processed_images :
            print("still processing ! can't move forward")
            self.ui.frame_slider.setValue(self.processed_images - 1)
            return
        
        ## show store detect
        if frame_num < self.processed_images:
            image, court = self.draw_images(frame_num)
            self.display_id_information(frame_num)
            self.show_image(image, self.video_scene, self.ui.Video_View)
            self.show_image(court, self.court_scene, self.ui.Virtual_Court)

            if self.display_window.isVisible():
                self.display_window.show_image(image, court)
            return
        
        ori_image=self.video_images[frame_num].copy()
        image=ori_image.copy()
 
        video_end = False
        if self.ui.frame_slider.value() == (self.total_images-3) :
            video_end =True
        show_track=True if self.ui.bbox_check.isChecked() else False
        self.timer.tic()
        BBox_data, image = self.pose2d.process_image(image, show_track, frame_id = frame_num)
        average_time = self.timer.toc()
       
        self.get_points_to_be_drawn(BBox_data, frame_num)

        if not show_track:
            image, court = self.draw_images(frame_num)

        if video_end:
            def writecsv(csvfile, revised_id):
                with open(csvfile, 'w',newline="") as file:
                    writer = csv.writer(file)
                    for _ in revised_id:
                        writer.writerows(revised_id)
            writecsv(self.db_path + 'data/rev.csv', self.revise_id)
            print("success save revised id")

        if show_fps:
            fps=1/max(average_time,0.00001)  
            self.ui.fps_label.setText("fps: {}".format(round(fps, 2)))
        self.show_image(image, self.video_scene, self.ui.Video_View)

    def show_BBox(self, frame_number):
        image = self.video_images[frame_number].copy()
        if self.ui.bbox_check.isChecked():

            filtered_df = self.player_points_df.loc[self.player_points_df['frame_number'] == frame_number]
            if not filtered_df.empty:
                for _, row in filtered_df.iterrows():
                    bbox = tuple(map(int, (row['Bbox_x'], row['Bbox_y'], row['Bbox_x'] + row['Bbox_w'], row['Bbox_y'] + row['Bbox_h'])))
                    cv2.rectangle(image, bbox[0:2], bbox[2:4], color=(154,250,0), thickness=2)
                    t_x, t_y = bbox[0:2]
                    h, w, _ = image.shape
                    t_x = max(10, t_x+10) if t_x < 0 else min(t_x + 10,w - 50)
                    t_y = max(10, t_y-5) if t_y < 0 else min(h - 20, t_y - 5)
                    image=cv2.putText(image, str(int(row['remapped_id'])), (t_x, t_y) , cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)

        return image
    
    def play_frame(self, start_num=0):
        for i in range(start_num, self.total_images):
            if not self.is_play:
                break
            if i > self.processed_images:
                self.processed_images = i
                self.analyze_frame()
            self.ui.frame_slider.setValue(i)
            # to the last frame ,stop playing
            if i == self.total_images - 1 and self.is_play:
                self.play_btn_clicked()
            # time.sleep(0.1)
            cv2.waitKey(15)
      
    def play_btn_clicked(self):
        if self.video_path == "":
            QMessageBox.warning(self, "無法開始播放", "請先讀取影片!")
            return
        self.is_play = not self.is_play
        if self.is_play:
            # self.ui.play_btn.setText("暫停")
            self.ui.play_btn.setIcon(self.pause_icon)
            self.play_frame(self.ui.frame_slider.value())
        else:
            # self.ui.play_btn.setText("播放")
            self.ui.play_btn.setIcon(self.play_icon)

    def save_video(self):
        if self.is_play:
            QMessageBox.warning(self, "儲存影片失敗", "請先停止播放影片!")
            return
        if self.saving_video:
            QMessageBox.warning(self, "儲存影片失敗", "請不要多次按下儲存影片按鈕!")
            return
        
        # save point data
    
        self.saving_video = True

        video_name = self.ui.video_label.text()
        video_size = (1920, 1080)
        fps = 30.0
        write_location = self.db_path + "output/"
        save_location = write_location + video_name

        video_writer = cv2.VideoWriter(save_location, cv2.VideoWriter_fourcc(*'mp4v'), fps, video_size)

        if not video_writer.isOpened():
            print("error while opening video writer!")
            return

    
        def resize_and_concatenate(real_image, virtual_image):
            target_height = 1080
            target_width = int(target_height / 2)
            
            new_shape = (target_width, target_height)

            conbined_image = np.concatenate((real_image, cv2.resize(virtual_image, new_shape)), axis=1)
            conbined_image = cv2.resize(conbined_image, (1920, 1080))

            return conbined_image

        for frame_number in range(int(self.player_points_df['frame_number'].max())):
            image, court = self.draw_images(frame_number)
            combined_image = resize_and_concatenate(image, court)
            video_writer.write(combined_image)

        video_writer.release()

        self.saving_video = False
        QMessageBox.information(self, "儲存影片", "影片儲存完成!")

    def get_points_to_be_drawn(self, full_data: dict, frame_number: int): #id

        non_assign_id = []
        all_real_point_dict = {}

        for person_id,BBox in full_data.items():
            try:
                # if new id appear in full data, catch key error
                if self.processed_images != 0 and self.id_mapping[person_id] is None:
                    continue
            except KeyError:
                pass
            x, y, w, h =BBox[0]
            circle_center = (x + w / 2, y + h)
            all_real_point_dict[person_id] = circle_center

        all_transformed_point_dict = self.get_virtual_court_point(all_real_point_dict)
        
        def point_is_inside(virtual_center):
            # filter inside player, output True if inside
            if virtual_center[0] < self.min_virtual_court_short_side or virtual_center[0] > self.max_virtual_court_short_side:
                return False
            return True
              
        def height_adjustment(virtual_center):
            # adjust out of bounds player to border of virtual court view
            if virtual_center[1] < 0:
                return (virtual_center[0], 0)
            elif virtual_center[1] > self.ui.Virtual_Court.height():
                return (virtual_center[0], self.ui.Virtual_Court.height() - 3)
            else:
                return virtual_center
            
        def get_team_side(real_center):
            # put calculate team side based on a * x + b
            # 0: blue ;1: green
            team_side = 0 if self.line_parameter == None or real_center[1] < self.line_parameter[0][0] * real_center[0] + self.line_parameter[0][1] else 1
            return team_side
        
        # first frame person id is remapped and record it remapped id list in id map 
        def record_id_map(remap_data):
            for person_id,remapped_id in remap_data.items():
                if person_id == None or remapped_id == None:
                    pass
                else:
                    self.id_map[remapped_id][0].append(person_id)

        # assign non-assigned person id a remapped id     
        def assign_id(person_id,start_index,end_index):
            for i in range(start_index , (end_index + 1)):
                if self.id_map[i][1] == False:
                    self.id_map[i][0].append(person_id)
                    self.id_map[i][1] = True
                    break
        
        def check_is_player(person_id):
            is_player = False

            for id_list in self.id_map.values():
                if person_id in id_list[0] and check_bbox_size(person_id):
                    is_player = True
                    break
            return is_player

        # check person id is in court or not          
        def check_player_id():
            self.id_mapping = {}
            for person_id in full_data.keys():
                if person_id not in self.id_mapping.keys():
                    virtual_center = all_transformed_point_dict[person_id]
                    if point_is_inside(virtual_center) :           
                        self.id_mapping[person_id] = -1
                    elif check_is_player(person_id):
                        self.id_mapping[person_id] = -2

        # mapping the person id who is in the court 
        def map_id():
            for person_id in self.id_mapping.keys():
                for remapp_id ,id_list in self.id_map.items():
                    if person_id in id_list[0] and id_list[1] == False:
                        self.id_mapping[person_id] = remapp_id
                        id_list[1] = True
                        break
                    # handle collisions
                    elif person_id in id_list[0] and id_list[1] == True:
                        try:
                            
                            for id in id_list[0]:
                                if id in self.id_mapping.keys():
                                    collision_person_id = id
                                    break

                            index1 = id_list.index(collision_person_id)
                            index2 = id_list.index(person_id)

                            if index1 < index2:
                                self.id_mapping[person_id] = -1
                                id_list[0].remove(person_id)
                            elif index1 > index2:
                                self.id_mapping[collision_person_id] = -1
                                id_list[0].remove(collision_person_id)

                        except ValueError:
                            pass

        def check_team_side():
            #team_switched_id = {person_id : [remapped_id ,age]}
            
            for person_id ,remapped_id in self.id_mapping.items():
                real_center = all_real_point_dict[person_id]
                if remapped_id > 0 and remapped_id < 7 and not get_team_side(real_center):
                    if person_id in self.blue_team_switched_id.keys():
                        self.blue_team_switched_id[person_id][1] += 1
                    else:
                        self.blue_team_switched_id[person_id] = [remapped_id ,1]
                elif remapped_id > 6  and remapped_id < 13 and get_team_side(real_center):
                    if person_id in self.green_team_switched_id.keys():
                        self.green_team_switched_id[person_id][1] += 1
                    else:
                        self.green_team_switched_id[person_id] = [remapped_id ,1]
                
                if remapped_id < 7 and get_team_side(real_center) and person_id in self.blue_team_switched_id.keys():
                    self.blue_team_switched_id[person_id][1] -= 1
                    if self.blue_team_switched_id[person_id][1] == 0 or  self.blue_team_switched_id[person_id][1] < 0:
                        self.blue_team_switched_id.pop(person_id)
                elif remapped_id > 6 and not get_team_side(real_center) and person_id in self.green_team_switched_id.keys():
                    self.green_team_switched_id[person_id][1] -= 1
                    if self.green_team_switched_id[person_id][1] == 0 or self.green_team_switched_id[person_id][1] < 0:
                        self.green_team_switched_id.pop(person_id)
        
        def change_team_side(team_switched_id):
            for person_id ,information in team_switched_id.copy().items():
                remapped_id = information[0]
                age = information[1]
                if age > 50:
                    self.id_mapping[person_id] = -1
                    self.id_map[remapped_id][0].remove(person_id)
                    self.id_map[remapped_id][1] = False
                    team_switched_id.pop(person_id)
                    print(self.id_map)
                    # self.is_play = False
                    
        def clear_id_map_status():
            for id_list in self.id_map.values():
                id_list[1] = False

        def check_bbox_size(person_id):
            is_player_size = True
            BBox = full_data[person_id]
            x, y, w, h =BBox[0]
            min_box_area = 5000
            if w*h < min_box_area:
                is_player_size = False
            return is_player_size

        if self.processed_images == 0:
            # first frame, check each point and see if it is inside the court
            transformed_point_dict = {}
            team_side_dict = {}
            for person_id in full_data.keys():
                real_center = all_real_point_dict[person_id]
                virtual_center = all_transformed_point_dict[person_id]
                if point_is_inside(virtual_center):
                    transformed_point_dict[person_id] = virtual_center
                    team_side_dict[person_id] = get_team_side(real_center)
        
            # use ** to unpack the dict and combine back to one dict
            # if key is in both dict, later one takes precedence
            record_id_map(self.remap_data(transformed_point_dict, team_side_dict))
            check_player_id()
            map_id()
            
        else:
            # after that, check if id is in id map or not
            clear_id_map_status()
            check_player_id()
            map_id()
            if len(self.blue_team_switched_id) != 0:
                change_team_side(self.blue_team_switched_id)
            if len(self.green_team_switched_id) != 0:
                change_team_side(self.green_team_switched_id)

            #search person id who is not assign remapped_id and assign a new remapped_id for it      
            for person_id , remapped_id in self.id_mapping.items():
                if remapped_id == -1:
                    try:
                        # non_assign_id.append(person_id)
                        real_center = all_real_point_dict[person_id]
                        if get_team_side(real_center):
                            assign_id(person_id,1,6)
                        else:
                            assign_id(person_id,7,12)
                    except KeyError:
                        pass
        
            check_team_side()
           
            # print(self.blue_team_switched_id)
            # print(self.green_team_switched_id)

        for person_id in self.id_mapping.keys():
            if person_id in full_data.keys() and self.id_mapping[person_id] !=-1:
                x, y, w, h = full_data[person_id][0]
                virtual_center = height_adjustment(all_transformed_point_dict[person_id])
                if self.id_mapping[person_id] < 7 :
                    self.player_points_df.loc[len(self.player_points_df)] = [int(frame_number), int(person_id), self.id_mapping[person_id], # frame_num, ID, remapped ID | ID_mapping
                                                                                    round(x, 2), round(y, 2), round(w, 2), round(h, 2), # bounding box x, y, w, h
                                                                                    round(virtual_center[0], 2), round(virtual_center[1], 2) # virtual court x and y
                                                                                    ]
        
        return

    def draw_images(self, frame_number, court = None):

        image = self.video_images[frame_number].copy()

        if court is None:
            court = self.blank_court.copy()

        if self.ui.line_check.isChecked():
            def show_line(image: np.ndarray): 
                # get points by y = ax + b
                _,w,_=image.shape
                point1 = (0, int(self.line_parameter[0][0] * 0 + self.line_parameter[0][1]))
                point2 = (w, int(self.line_parameter[0][0] * w + self.line_parameter[0][1]))

                image = cv2.line(image, point1, point2, color=(0, 255, 0), thickness=2)
                
                return image
            image = show_line(image)
        
        color_palette = [(252, 177, 3), (0, 255, 0),(0,0,255)] 
        radius = 10

        filtered_data_df = self.player_points_df.loc[self.player_points_df['frame_number'] == frame_number]
        self.tracking_line = []
        if not filtered_data_df.empty:

            indexed_filtering = filtered_data_df.reset_index().drop('index', axis=1)
            for index, row in indexed_filtering.iterrows():
                # draw on real image
                real_center = (int(row['Bbox_x'] + row['Bbox_w'] / 2), int(row['Bbox_y'] + row['Bbox_h']))
                real_text_pos = (real_center[0] - 10, real_center[1]+7)
                display_id = row['remapped_id']
                # team side: 0: blue team; 1: green team; 2: not assigned team
                if display_id > 6 : 
                    team_side = 1 
                    display_id -= 6
                elif display_id == -1:
                    team_side = 2
                    display_id = row['object_id']
                    if display_id > 9 :
                        real_text_pos = (real_text_pos[0] - 5, real_text_pos[1])
                else :
                    team_side = 0

                image = cv2.circle(image, center=real_center, radius=radius, color=color_palette[team_side], thickness=-1)
                image = cv2.putText(image, '%d' % (display_id), real_text_pos, cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), thickness=2)

                # draw on virtual plane
                virtual_center = (int(row['virtual_x']), int(row['virtual_y']))
                
                if display_id > 9 : 
                    virtual_text_pos = (virtual_center[0] - 16, virtual_center[1]+7)
                else :
                    virtual_text_pos = (virtual_center[0] - 8, virtual_center[1]+7)

                if (display_id == self.ui.track_first_player_number.value()) or (display_id == self.ui.track_second_player_number.value()) or (display_id == self.ui.track_third_player_number.value()):
                    if display_id != 0:
                        self.tracking_line.append([virtual_center,virtual_text_pos,display_id])
                
                # print(self.show_tracking)
                court = cv2.circle(court, center=virtual_center, radius=radius, color=color_palette[team_side], thickness=-1)
                court = cv2.putText(court, '%d' % (display_id), virtual_text_pos, cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), thickness=2)
            # print(self.tracking_line)
            if self.show_tracking and len(self.tracking_line) == 3:
                for i in range(len(self.tracking_line)):
                    court = cv2.line(court, self.tracking_line[i][0], self.tracking_line[(i+1)%3][0], (0, 0, 255), thickness=2)
                for i in range(len(self.tracking_line)):
                    color_palette = [(252, 177, 3), (0, 255, 0),(0,0,255)] 
                    radius = 10
                    court = cv2.circle(court, center=self.tracking_line[i][0], radius=radius, color=color_palette[0], thickness=-1)
                    court = cv2.putText(court, '%d' % (self.tracking_line[i][2]), self.tracking_line[i][1], cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), thickness=2) 
            
            elif self.show_tracking and len(self.tracking_line) == 2 :
               
                court = cv2.line(court, self.tracking_line[0][0], self.tracking_line[1][0], (0, 0, 255), thickness=2)
                for i in range(len(self.tracking_line)):
                    color_palette = [(252, 177, 3), (0, 255, 0),(0,0,255)] 
                    radius = 10
                    court = cv2.circle(court, center=self.tracking_line[i][0], radius=radius, color=color_palette[0], thickness=-1)
                    court = cv2.putText(court, '%d' % (self.tracking_line[i][2]), self.tracking_line[i][1], cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), thickness=2)   
        if self.serving_reception_data is not None:
            serving_point, reception_point = self.serving_reception_data.values()
            court = cv2.line(court, serving_point, reception_point, (0, 0, 255), thickness = 2)

        return image, court

    def start_draw_line_thread(self):
        if not self.is_threading and self.total_images > 0 and not self.is_play:
            self.is_threading=True
            QMessageBox.warning(self, "畫線方法", "在兩根球網柱子底部各點一個點!\nLeft click at the bottom of both post!")
            self.line_thread = CVLineThread(self.video_images[0])
            self.line_thread.emit_signal.connect(self.set_line)
            self.line_thread.start()

    def set_line(self, img_line, save = True):
        self.close_thread(self.line_thread)
        if len(img_line)<=1 :
            QMessageBox.warning(self, "畫線失敗", "請在畫面上點取兩點!")
            return
        x1, y1 = img_line[0]
        x2, y2 = img_line[1]
        a = (y1 - y2) / (x1 - x2)
        b = y1 - a * x1

        self.line_parameter = [(a, b)] # y = ax + b, (a, b)
        if save:
            with open(self.db_path + 'data/line_parameter.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self.line_parameter)
    
    def draw_court(self, horizontal_view = False):
        # Please reference VolleyballCourt.png for what is going on in this function
        width = self.ui.Virtual_Court.height()
        height = self.ui.Virtual_Court.width()

        self.blank_court = np.full(shape = (width, height, 3), fill_value=255, dtype=np.uint8)

        borderline_thickness = 2
        outside_color = (94, 191, 88)
        inside_color = (82, 189, 250)
        middle_line_color = (214, 9, 180) 
        borderline_color = (255, 255, 255)

        chunk_line_color = borderline_color # (214, 9, 180) 
        chunk_line_thickness = 2
        short_side_offset = 0 # things are weird on QT, just fix with offset
        long_side_offset = 0 # I hate Qt, offset is sometimes required but sometimes not
        
        # calculate border points
        long_side_ratio = [4, 6, 3, 3, 6, 4] # from left to right, in meters
        short_side_ratio = [2, 9, 2] # from top to bottom, in meters
        short_side_sum = np.sum(short_side_ratio)
        long_side_sum = np.sum(long_side_ratio)

        def get_court_border_points():

            # order of points:
            # 0 ~ 3: outer border, counter-clockwise starting from top-left, long-side first
            # 4 , 5: middle line, left then right
            # 6 , 7: 3m line close to top of window, left then right
            # 8 , 9: 3m line close to bottom of window, left then right
            court_points = [] 

            # border points
            unit_distance = [int(height / short_side_sum), int(width / long_side_sum)]
            # top-left
            court_points.append(tuple((unit_distance[0] * short_side_ratio[0] - short_side_offset, unit_distance[1] * long_side_ratio[0] - long_side_offset)))
            # top-right
            court_points.append(tuple((unit_distance[0] * short_side_ratio[0] - short_side_offset, width - unit_distance[1] * long_side_ratio[0])))
            # bottom-right
            court_points.append(tuple((height - unit_distance[0] * short_side_ratio[0], width - unit_distance[1] * long_side_ratio[0])))
            # bottom-left
            court_points.append(tuple((height - unit_distance[0] * short_side_ratio[0], unit_distance[1] * long_side_ratio[0] - long_side_offset)))

            # middle line
            middle_line_width = int(width / 2)
            # middle line, left point
            court_points.append(tuple((unit_distance[0] * short_side_ratio[0] - short_side_offset, middle_line_width)))
            # middle line, right point
            court_points.append(tuple((height - unit_distance[0] * short_side_ratio[0], middle_line_width)))

            # 3m line, top one
            three_meter_distance = int(unit_distance[1] * np.sum(long_side_ratio[2])) # extend to 3m, how did I get mixed up with short side in unit distance?
            # left point
            court_points.append(tuple((unit_distance[0] * short_side_ratio[0] - short_side_offset, middle_line_width - three_meter_distance)))
            # right point
            court_points.append(tuple((height - unit_distance[0] * short_side_ratio[0], middle_line_width - three_meter_distance)))

            # 3m line, bottom one
            # left point
            court_points.append(tuple((unit_distance[0] * short_side_ratio[0] - short_side_offset, middle_line_width + three_meter_distance)))
            # right point
            court_points.append(tuple((height - unit_distance[0] * short_side_ratio[0], middle_line_width + three_meter_distance)))

            return court_points

        court_points = get_court_border_points()
        
        # fill the outside
        self.blank_court = cv2.rectangle(self.blank_court, (0, 0), (height, width), outside_color, -1)
        
        # inside court
        self.blank_court = cv2.rectangle(self.blank_court, (court_points[0][0], court_points[0][1]), (court_points[2][0], court_points[2][1]), inside_color, -1)

        # outer border
        # top-left to top-right
        self.blank_court = cv2.line(self.blank_court, court_points[0], court_points[1], borderline_color, borderline_thickness)
        # top-right to bottom-right
        self.blank_court = cv2.line(self.blank_court, court_points[1], court_points[2], borderline_color, borderline_thickness)
        # bottom-right to bottom-left
        self.blank_court = cv2.line(self.blank_court, court_points[2], court_points[3], borderline_color, borderline_thickness)
        # top-left to bottom-left
        self.blank_court = cv2.line(self.blank_court, court_points[0], court_points[3], borderline_color, borderline_thickness)
        # middle line
        self.blank_court = cv2.line(self.blank_court, court_points[4], court_points[5], middle_line_color, borderline_thickness)
        # 3m line, top one
        self.blank_court = cv2.line(self.blank_court, court_points[6], court_points[7], borderline_color, borderline_thickness)
        # 3m line, bottom one
        self.blank_court = cv2.line(self.blank_court, court_points[8], court_points[9], borderline_color, borderline_thickness)

        # print(court_points)
        # divide the court into (3m x 3m) chunks with purple lines
        def get_chunk_points():
            # 0, 1: 2 points on the top side
            # 2, 3: 2 points on the bottom side
            # 4, 5: 2 points in top court, dividing between 3m line and the rest of court
            # 6, 7: 2 points in bottom court, dividing between 3m line and the rest of court

            chunk_points = []

            short_side_one_chunk_length = int((court_points[3][0] - court_points[1][0]) / 3)
            long_side_top_middle_point = int((court_points[0][1] + court_points[6][1]) / 2)
            long_side_bottom_middle_point = int((court_points[1][1] + court_points[8][1]) / 2)

            # 2 points on the top side
            chunk_points.append(tuple((court_points[0][0] + short_side_one_chunk_length, court_points[0][1])))
            chunk_points.append(tuple((court_points[3][0] - short_side_one_chunk_length, court_points[3][1])))

            # 2 points on the bottom side
            chunk_points.append(tuple((court_points[1][0] + short_side_one_chunk_length, court_points[1][1])))
            chunk_points.append(tuple((court_points[2][0] - short_side_one_chunk_length, court_points[2][1])))

            # 2 points in top court, dividing between 3m line and the rest of court
            chunk_points.append(tuple((court_points[0][0], long_side_top_middle_point)))
            chunk_points.append(tuple((court_points[3][0], long_side_top_middle_point)))

            # 2 points in bottom court, dividing between 3m line and the rest of court
            chunk_points.append(tuple((court_points[1][0], long_side_bottom_middle_point)))
            chunk_points.append(tuple((court_points[2][0], long_side_bottom_middle_point)))

            return chunk_points
        
        chunk_points = get_chunk_points()
        # top to bottom, 2 lines
        self.blank_court = cv2.line(self.blank_court, chunk_points[0], chunk_points[2], chunk_line_color, chunk_line_thickness)
        self.blank_court = cv2.line(self.blank_court, chunk_points[1], chunk_points[3], chunk_line_color, chunk_line_thickness)
        # left to right, 1 line on top, 1 line at bottom
        self.blank_court = cv2.line(self.blank_court, chunk_points[4], chunk_points[5], chunk_line_color, chunk_line_thickness)
        self.blank_court = cv2.line(self.blank_court, chunk_points[6], chunk_points[7], chunk_line_color, chunk_line_thickness)

        if horizontal_view:
            self.blank_court = np.transpose(self.blank_court, axes=[1, 0, 2]) # turn the vertical court back to horizontal
        
        self.show_image(self.blank_court, self.court_scene, self.ui.Virtual_Court)

        return court_points

    def draw_perspective_points_start(self):
        if not self.is_threading and len(self.video_images)>0 and not self.is_play:
            self.is_threading=True
            self.points_thread = CVThread(self.video_images[0])
            self.points_thread.emit_signal.connect(self.calculate_perspective_matrix)
            QMessageBox.warning(self, "畫點順序", "從短邊開始順時針新增點!\nStarting from short side, please add the points in clockwise order!")
            self.points_thread.start()
           
    def calculate_perspective_matrix(self, point_list:np.ndarray, save = True):
        try:
            point = point_list[3]
        except IndexError:
            QMessageBox.warning(self, "球場繪製出錯", "請點選完整的球場!")
            self.is_threading = False
            return

        input_pts = np.float32(point_list[0:4])

        output_pts = np.float32([[self.min_virtual_court_short_side, self.min_virtual_court_long_side],
                                [self.max_virtual_court_short_side, self.min_virtual_court_long_side],
                                [self.max_virtual_court_short_side, self.max_virtual_court_long_side],
                                [self.min_virtual_court_short_side, self.max_virtual_court_long_side]])
        
        self.perspective_matrix = cv2.getPerspectiveTransform(input_pts,output_pts)
        if save:
            df = pd.DataFrame(self.perspective_matrix)
            df.to_csv(self.db_path + "data/matrix.csv", header=False, index=False)

    def get_virtual_court_point(self, point_dict: dict):

        # Convert original image points to homogeneous coordinates
            
        original_points_homogeneous = np.array([[point[0], point[1], 1] for point in point_dict.values()]).T

        # Apply the perspective transformation matrix to the original points to get the transformed points
        if self.perspective_matrix is not None:
            transformed_points_homogeneous = np.dot(self.perspective_matrix, original_points_homogeneous)

        # Convert the transformed points back to 2D coordinates
        transformed_points_homogeneous /= transformed_points_homogeneous[2]
        transformed_points = transformed_points_homogeneous[:2].T

        transformed_points = np.int32(transformed_points) 
        
        transformed_dict = {}
        for index, person_id in enumerate(point_dict):
            transformed_dict[person_id] = (transformed_points[index][0], transformed_points[index][1])
        
        return transformed_dict
        
    def start_tracking(self):

        self.tracking_court = self.blank_court.copy()
        self.show_tracking = True
        
        self.ui.track_first_player_number.setDisabled(True)
        self.ui.track_second_player_number.setDisabled(True)
        self.ui.track_third_player_number.setDisabled(True)

        self.tracklet = []

        player_num = self.ui.tracking_player_selector.currentIndex()
        if player_num == 0:
            self.tracking_player_number = self.ui.track_first_player_number.value()
        elif player_num == 1:
            self.tracking_player_number = self.ui.track_second_player_number.value()
        elif player_num == 2:
            self.tracking_player_number = self.ui.track_third_player_number.value()
        
        self.show_image(self.tracking_court, self.court_scene, self.ui.Virtual_Court)

        # Timer for show tracking
        try :
            self.tracking_timer = ShowTrackThread(self.player_points_df['frame_number'].max())
            self.tracking_timer.timeout_signal.connect(self.show_player_track)
            self.tracking_timer.finished_signal.connect(self.stop_tracking)
            self.tracking_timer.start()  

            self.tracking_clock=Timer()

            self.x_line.set_data([], [])
            self.y_line.set_data([], [])

        except Exception as e:
            print(e)

    def show_player_track(self, frame_num):
        
        self.tracking_clock.tic()

        self.ui.frame_slider.setValue(frame_num)
         
        filtered_data_df = self.player_points_df.loc[(self.player_points_df['frame_number'] == frame_num) & (self.player_points_df['remapped_id'] == self.tracking_player_number)]
       
        if filtered_data_df.empty:
            video_image, _ = self.draw_images(frame_num)
            self.show_image(video_image, self.video_scene, self.ui.Video_View)
            return
        else:
            bbox = tuple(map(int, (filtered_data_df['Bbox_x'].values[0], filtered_data_df['Bbox_y'].values[0], filtered_data_df['Bbox_w'].values[0], filtered_data_df['Bbox_h'].values[0])))
            person_id = int(filtered_data_df['remapped_id'].values[0])
            virtual_center = tuple(map(int, (filtered_data_df['virtual_x'].values[0], filtered_data_df['virtual_y'].values[0])))
        
            if len(self.tracklet) <0 or len(self.tracklet) < 31:
                self.tracklet.append(virtual_center)
            
            elif len(self.tracklet) > 30:
                self.tracklet.pop(0)
            
            if self.ui.virtual_point_figure_checkbox.isChecked():
                self.chart_add_point(frame_num, virtual_center[0], virtual_center[1])
            else:
                self.chart_add_point(frame_num, bbox[0] + bbox[2] / 2, bbox[1] + bbox[3])
 
        # self.tracking_court = cv2.circle(self.tracking_court, center=virtual_center, radius=2, color=(255, 0, 255), thickness=-1)
        tracking_court = self.tracking_court.copy()
        prev_v_center = None
        for v_center in self.tracklet:
            curr_v_center = v_center
            if prev_v_center is None:
                prev_v_center = v_center
            smooth_x = self.one_euro_f(v_center[0],prev_v_center[0])
            smooth_y = self.one_euro_f(v_center[1],prev_v_center[1])
            curr_v_center = (int(smooth_x), int(smooth_y))
            tracking_court = cv2.line(tracking_court, prev_v_center , curr_v_center, color=(255, 0, 255), thickness=2)
            prev_v_center = curr_v_center
        # tracking_court = self.tracking_court.copy()
        video_image, tracking_court = self.draw_images(frame_num, tracking_court)

        pose_point = get_pose_points(video_image, bbox, self.pose2d.pose)
        skeleton_image = draw_points_and_skeleton(video_image, pose_point[0], joints_dict()['coco']['skeleton'], person_index = person_id
                                                  ,points_color_palette='gist_rainbow', skeleton_palette_samples='jet', points_palette_samples=10
                                                  , confidence_threshold=0.3)

        self.show_image(skeleton_image, self.video_scene, self.ui.Video_View)
        self.show_image(tracking_court, self.court_scene, self.ui.Virtual_Court)

        if self.display_window.isVisible():
            self.display_window.show_image(skeleton_image, tracking_court)

    def linear_interpolation(self, point1, point2, weight):
        """
        Linear interpolation between two points.
        Args:
            point1 (tuple): Coordinates of the first point (x, y).
            point2 (tuple): Coordinates of the second point (x, y).
            weight (float): Weight for interpolation (0.0 to 1.0).p
        Returns:
            tuple: Interpolated coordinates (x, y).
        """
        x_interpolated = point1[0] + (point2[0] - point1[0]) * weight
        y_interpolated = point1[1] + (point2[1] - point1[1]) * weight
        return (x_interpolated, y_interpolated)

    def stop_tracking(self):
        self.close_thread(self.tracking_timer)
        self.show_tracking = False
        self.ui.track_first_player_number.setEnabled(True)
        self.ui.track_second_player_number.setEnabled(True)
        self.ui.track_third_player_number.setEnabled(True)
        
    def init_figure(self, frame_count):
        if self.ui.virtual_point_figure_checkbox.isChecked():
            width = self.ui.Virtual_Court.width()
            height = self.ui.Virtual_Court.height()
            title = "Player Location in Virtual Court"
        else:
            height, width, _ = self.video_images[0].shape
            title = "Player Location in Real Image"

        figure_scene = QGraphicsScene()
        self.ui.chart_view.setScene(figure_scene)
        chart_figure = Figure(figsize=(7.7, 2.5))

        self.chart_canvas = FigureCanvas(chart_figure)
        figure_scene.addWidget(self.chart_canvas)

        color_x = 'tab:red'
        color_y = 'tab:blue'
        self.x_line = Line2D([], [], color = color_x, linestyle='-', marker='', markersize=0, linewidth=1)
        self.x_line.set_label("X")
        self.y_line = Line2D([], [], color = color_y, linestyle='-', marker='', markersize=0, linewidth=1)
        self.y_line.set_label("Y")

        self.x_ax = chart_figure.add_subplot(111)
        self.x_ax.add_line(self.x_line)
        self.x_ax.set_xlabel("Frames")
        self.x_ax.set_xlim(0, frame_count)
        self.x_ax.axes.set_xticks(np.arange(0, (frame_count + 1), 100))
        self.x_ax.set_ylim(0, width)
        self.x_ax.set_ylabel("X", rotation = 0, color = color_x)
        self.x_ax.tick_params(axis="y", labelcolor = color_x)
        self.x_ax.axes.set_yticks(np.arange(0, (width + 1), width / 8))
        self.x_ax.set_title(title)

        self.y_ax = self.x_ax.twinx()
        self.y_ax.add_line(self.y_line)
        self.y_ax.set_ylim(0, height)
        self.y_ax.set_ylabel("Y", rotation = 0, color = color_y)
        self.y_ax.tick_params(axis="y", labelcolor = color_y)
        self.y_ax.axes.set_yticks(np.arange(0, (height + 1), height / 8))
        self.y_ax.axes.invert_yaxis()

        chart_figure.legend(loc = "lower right")
        chart_figure.tight_layout()

    def chart_add_point(self, frame_number, x, y):
        existing_frame_data, existing_x_data = self.x_line.get_xdata(), self.x_line.get_ydata()
        existing_y_data = self.y_line.get_ydata()

        updated_frame_data = list(existing_frame_data)
        updated_x_data = list(existing_x_data)
        updated_y_data = list(existing_y_data)

        updated_frame_data.append(frame_number)
        updated_x_data.append(x)
        updated_y_data.append(y)

        # print("frame: ", updated_frame_data)
        # print("-----------")
        # print("x: ", updated_x_data)
        # print("-----------")
        # print("y: ", updated_y_data)
        # print("-----------")

        self.x_line.set_data(updated_frame_data, updated_x_data)
        self.y_line.set_data(updated_frame_data, updated_y_data)

        self.chart_canvas.draw_idle()

    def load_csv(self):
        if self.total_images == 0:
            QMessageBox.warning(self, "讀取失敗", "請先讀取影片!")
            return
        video_name = self.ui.video_label.text().split(".")[0]

        all_points_filename = video_name + "_AllPoints.csv"
        all_points_full_path = self.db_path + "data/" + all_points_filename
        
        if not os.path.exists(all_points_full_path):
            QMessageBox.warning(self, "讀取失敗", "CSV 檔\n" + all_points_filename + "\n不存在於\n" + self.db_path + "data/")
            return

        self.player_points_df = pd.read_csv(all_points_full_path)

        self.processed_images = self.player_points_df['frame_number'].max()
        self.analyze_frame()

        QMessageBox.information(self, "讀取成功", "CSV 檔讀取完成")

    def save_csv(self):
        if self.processed_images == -1:
            QMessageBox.warning(self, "儲存失敗", "沒有已經處理好的影像!")
            return
        
        video_name = self.ui.video_label.text().split(".")[0]

        all_points_filename = video_name + "_AllPoints.csv"
        all_points_full_path = self.db_path + "data/" + all_points_filename

        self.player_points_df.to_csv(all_points_full_path, index=False)

        QMessageBox.information(self, "儲存成功", "CSV 檔儲存完成")

    def draw_figure(self):
        if self.total_images == 0:
            QMessageBox.warning(self, "繪製失敗", "請先讀取影片!")
            return

        player_number = self.ui.figure_player_number.value() + 6 * self.ui.figure_team_side_selector.currentIndex()  # 0: 藍隊; 1: 綠隊

        filtered_points = self.player_points_df.loc[(self.player_points_df['remapped_id'] == player_number)]  
        if filtered_points.empty:
            return

        frame_data = filtered_points["frame_number"].values

        if self.ui.virtual_point_figure_checkbox.isChecked():
            x_data = filtered_points["virtual_x"].values
            y_data = filtered_points["virtual_y"].values
        else:
            x_data = filtered_points["Bbox_x"].values + (filtered_points["Bbox_w"].values / 2)
            y_data = filtered_points["Bbox_y"].values + filtered_points["Bbox_h"].values

        self.x_line.set_data(frame_data, x_data)
        self.y_line.set_data(frame_data, y_data)

        self.chart_canvas.draw()

    def swap_figure_y_limit(self):
        # FIXME: if swap figure before SOT complete, weird bugs happen

        try:
            self.x_ax.get_xlabel()
        except AttributeError:
            return

        if self.ui.virtual_point_figure_checkbox.isChecked():
            width = self.ui.Virtual_Court.width()
            height = self.ui.Virtual_Court.height()
            title = "Player Location in Virtual Court"
        else:
            height, width, _ = self.video_images[0].shape
            title = "Player Location in Real Image"

        self.x_ax.set_title(title)

        self.x_ax.set_ylim(0, width)
        self.y_ax.set_ylim(0, height)

        self.x_ax.axes.set_yticks(np.arange(0, (width + 1), width / 8))
        self.y_ax.axes.set_yticks(np.arange(0, (height + 1), height / 8))

        self.y_ax.axes.invert_yaxis()

        self.draw_figure()

    def revise_btn_clicked(self):
             
        def revise_id_csv(id1,id2,start_frame_num,end_frame_num):

            for frame_number in range(start_frame_num,(end_frame_num+1)):
                df_copy = self.player_points_df.copy()
                # 找到要交換的行的條件
                condition_1 = (df_copy['frame_number'] == frame_number) & (df_copy['remapped_id'] == id1)
                condition_2 = (df_copy['frame_number'] == frame_number) & (df_copy['remapped_id'] == id2)
                df_copy.loc[condition_1, 'remapped_id'] = -1
                df_copy.loc[condition_2, 'remapped_id'] = id1
                condition_3 = (df_copy['frame_number'] == frame_number) & (df_copy['remapped_id'] == -1)
                # print(condition_3)
                df_copy.loc[condition_3, 'remapped_id'] = id2

                # 將原始 DataFrame 替換為複製
                self.player_points_df = df_copy


        before_revision_id = self.ui.before_revision_spinbox.value()
        after_revision_id = self.ui.after_revision_spinbox.value()

        if self.ui.function_combo_box.currentIndex() == 0:
            if after_revision_id == 0 or before_revision_id == 0:
                QMessageBox.information(self, "ID 選擇錯誤", "ID不可為0")
            else:
                #交換ID
                start_frame_num = self.ui.start_frame_num_spinbox.value()
                end_frame_num = self.ui.end_frame_num_spinbox.value()
                revise_id_csv(before_revision_id,after_revision_id,start_frame_num,end_frame_num)
                before_revision_key = [key for key, value in self.id_mapping.items() if value == before_revision_id]
                after_revision_key = [key for key, value in self.id_mapping.items() if value == after_revision_id]

                if len(before_revision_key) != 0:
                    if before_revision_key[0] in self.id_mapping.keys():
                        self.id_mapping[before_revision_key[0]] = after_revision_id

                    if before_revision_id in self.id_map.keys() and before_revision_key[0] in self.id_map[before_revision_id][0]:
                        self.id_map[before_revision_id][0].remove(before_revision_key[0])
                        self.id_map[after_revision_id][0].append(before_revision_key[0])

                if len(after_revision_key) != 0:
                    if after_revision_key[0] in self.id_mapping.keys():
                        self.id_mapping[after_revision_key[0]] = before_revision_id
                    if after_revision_id in self.id_map.keys() and after_revision_key[0] in self.id_map[after_revision_id][0]:
                        self.id_map[after_revision_id][0].remove(after_revision_key[0])
                        self.id_map[before_revision_id][0].append(after_revision_key[0])
                        
                #顯示交換ID的資訊
                display_id = after_revision_id - 6 if after_revision_id > 6 else after_revision_id
                QMessageBox.information(self, "修改成功", "已經將藍隊的 " + str(before_revision_id) + " 號和藍隊的: "+ str(display_id) + " 號交換回來了!")
    
    def remap_data(self, transformed_point_dict: dict, above_line_dict: dict, verbose = False):
        """
        Remap person id to 1 ~ 6 based id
        ## Args:
            transformed_point_dict: player data on virtual court in one frame\n
            above_line_dict: used to distinguish two teams
        ## Returns:
            one dict that consists of (original id: remapped id) on the both sides

        this function is called every frame
        """

        # Define a custom sorting key that accepts the starting_point as an argument
        def sort_key_clockwise(starting_point):
            def calculate_angle(item):
                if verbose:
                    print(item)
                x, y = item[1]
                # Calculate the angle using arctan2 relative to the starting_point
                angle = np.arctan2(y - starting_point[1], x - starting_point[0])
                # Return the angle as the sorting key
                return angle
            return calculate_angle
        
        left_side_data = {}
        right_side_data = {}
        
        for person_id in above_line_dict:
            if above_line_dict[person_id] == False:
                right_side_data[person_id] = transformed_point_dict[person_id]
            else:
                left_side_data[person_id] = transformed_point_dict[person_id]

        # ------------ left side ---------------

        # Sort the data using the custom sorting key
        top_left_point = min(left_side_data.values(), key = lambda point: (point[0], point[1]))
        if verbose:
            print(top_left_point)
        sorted_data = sorted(left_side_data.items(), key=sort_key_clockwise(starting_point = top_left_point))

        # Create a mapping from original IDs to new IDs
        left_side_id_mapping = {original_id: new_id for new_id, (original_id, _) in enumerate(sorted_data, start=1)}
        
        # # ------------ right side ---------------
        # # Sort the data using the custom sorting key
        # top_left_point = min(right_side_data.values(), key = lambda point: (point[0], point[1]))
        # if verbose:
        #     print(top_left_point)
        # sorted_data = sorted(right_side_data.items(), key=sort_key_clockwise(starting_point = top_left_point))

        # # Create a mapping from original IDs to new IDs
        # right_side_id_mapping = {original_id: new_id for new_id, (original_id, _) in enumerate(sorted_data, start=7)}

        # use ** to unpack the dict and combine back to one dict
        # if key is in both dict, later one takes precedence
        full_id_mapping = { **left_side_id_mapping}

        if verbose:
            print("-------------------------------------\n")
            # print(right_side_id_mapping)
            # print("\n")
            print(left_side_id_mapping)
            print("\n")
            print(full_id_mapping)
            print("-------------------------------------\n")
        return full_id_mapping
            
    def display_id_information (self,frame_number):
        green_team_id_information = ""
        blue_team_id_information = ""
        not_assigned_id_information = ""
        id_mapped_dict = {}
        filtered_data_df = self.player_points_df.loc[self.player_points_df['frame_number'] == frame_number]

        if not filtered_data_df.empty:

            indexed_filtering = filtered_data_df.reset_index().drop('index', axis=1)
            for index, row in indexed_filtering.iterrows():
                remapped_id = int(row['remapped_id'])
                object_id = int(row['object_id'])
                # team side: 0: blue team; 1: green team; 2: not assigned team
                id_mapped_dict[remapped_id] = object_id
            
            for i in range(1,13):
                if i == 3 or i == 5 :
                    blue_team_id_information += "\n"
                if i == 9 or i == 11:
                    green_team_id_information += "\n"
                if i < 7:
                    if i in id_mapped_dict.keys():
                        if id_mapped_dict[i] > 9:
                            blue_team_id_information += f"{i}   : {id_mapped_dict[i]}       ,"
                        else :
                            blue_team_id_information += f"{i}   : {id_mapped_dict[i]}         ,"
                    else:
                        blue_team_id_information += f"{i}   : None ,"
                else:
                    if i in id_mapped_dict.keys():
                        if id_mapped_dict[i] > 9 :
                            green_team_id_information += f"{(i-6)}   : {id_mapped_dict[i]}       ,"
                        else :
                            green_team_id_information += f"{(i-6)}   : {id_mapped_dict[i]}         ,"
                    else:
                        green_team_id_information += f"{(i-6)}   : None ,"
                
            for remapped_id , object_id in id_mapped_dict.items():
                if remapped_id == -1:
                    not_assigned_id_information += f" {object_id} ,"

            if len(not_assigned_id_information) == 0:
                not_assigned_id_information += "None"
            # else :
            #     self.is_play = not self.is_play
            #     self.ui.play_btn.setIcon(self.pause_icon)
            self.ui.green_team_player_id_label.setText(green_team_id_information)
            self.ui.blue_team_player_id_label.setText(blue_team_id_information)
            self.ui.not_assigned_player_id_label.setText(not_assigned_id_information)

    def show_display_window(self):
        if self.total_images == 0:
            return
        frame_num = self.ui.frame_slider.value()
        image, court = self.draw_images(frame_num)

        self.display_window.show_image(image, court)
        self.display_window.show()

    def is_not_player(self):

        try :
            remapped_id = self.ui.id_box.value()
            if remapped_id in self.id_map.keys() :
                person_id = [key for key, value in self.id_mapping.items() if value == remapped_id]
                person_id = person_id[0]
                self.id_map[remapped_id][0].remove(person_id)
                self.id_map[remapped_id][1] = False
                self.id_mapping.pop(person_id)

        except ValueError:
            pass



if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())