
from enum import Enum
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import cv2
import numpy
class DataType(Enum):
    DEFAULT = {"name": "default", "tips": "", "filter": ""}
    IMAGE = {"name": "image", "tips": "",
             "filter": "Image files (*.jpeg *.png *.tiff *.psd *.pdf *.eps *.gif)"}
    VIDEO = {"name": "video", "tips": "",
             "filter": "Video files ( *.WEBM *.MPG *.MP2 *.MPEG *.MPE *.MPV *.OGG *.MP4 *.M4P *.M4V *.AVI *.WMV *.MOV *.QT *.FLV *.SWF *.MKV)"}
    CSV = {"name": "csv", "tips": "",
           "filter": "Video files (*.csv)"}
    FOLDER = {"name": "folder", "tips": "", "filter": ""}




def video_to_frame(input_video_path):
    video_images = []
    vidcap = cv2.VideoCapture(input_video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success, image = vidcap.read()
    frame_counter = 0
    count = 0
    while success:
        # load in even number frame only
        if not frame_counter & 1:
            video_images.append(image)
            count += 1
        success, image = vidcap.read()
        frame_counter += 1
    vidcap.release()
    fps = int(fps) >> 1
    # set image count labels
    return video_images, fps, count
def is_in_polygon(p_range,point):
    x,y=point
    point = Point(x,y)
    polygon = Polygon(p_range)
    return polygon.contains(point)

# image to jpg example
# def save_jpg(row, dir_path):
#     file_name = f"{row['key.segment_context_name']};{row['key.frame_timestamp_micros']};{row['key.camera_name']}"
#     with open(f"{dir_path}/{file_name}.jpg",'wb') as f:
#         f.write(row['[CameraImageComponent].image'])
#     with open(f"{dir_path}/{file_name}.txt",'wb') as f:
#         pass

# row in data frame to txt example
# def save_txt(row, dir_path):
#     file_name = f"{row['key.segment_context_name']};{row['key.frame_timestamp_micros']};{row['key.camera_name']}"
#     heigh = 1280
#     width = 1920
#     x = row['[CameraBoxComponent].box.center.x'] / width
#     y = row['[CameraBoxComponent].box.center.y'] / heigh
#     x_size = row['[CameraBoxComponent].box.size.x'] / width
#     y_size = row['[CameraBoxComponent].box.size.y'] / heigh
#     object_type = row['[CameraBoxComponent].type']
#     lines = [
#         f"{object_type} {x} {y} {x_size} {y_size}\n"
#     ]
#     with open(f"{dir_path}/{file_name}.txt",'a') as f:
#         f.writelines(lines)