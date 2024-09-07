from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap

import numpy as np
import cv2

class DisplayWindow(QMainWindow):
    def __init__(self, parent = None):
        self.window_width = 1920        
        self.window_height = 1080

        super().__init__(parent)
        self.setWindowTitle("Display")
        # Set the size of the window
        self.setGeometry(100, 100, self.window_width, self.window_height)  # (x, y, width, height)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.hide()

        # Create a QGraphicsView widget
        self.view = QGraphicsView(self)
        self.view.setGeometry(0, 0, self.window_width, self.window_height)  # (x, y, width, height)
        self.setCentralWidget(self.view)
        # Set scroll bars to always be off
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Create a QGraphicsScene and set it on the view
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, self.window_width, self.window_height)  # (x, y, width, height)
        self.view.setScene(self.scene)

    def show_image(self, image: np.ndarray, court: np.ndarray):
        self.scene.clear()
        
        def resize_and_concatenate(real_image, virtual_image):
            target_height = 1080
            target_width = int(target_height / 2)
            
            new_shape = (target_width, target_height)

            combined_image = np.concatenate((real_image, cv2.resize(virtual_image, new_shape)), axis=1)
            combined_image = cv2.resize(combined_image, (self.window_width, self.window_height))

            return combined_image

        image = resize_and_concatenate(image, court)

        w,h = image.shape[1],image.shape[0]
        int
        bytesPerline = 3 * w
        qImg = QImage(image, w, h, bytesPerline, QImage.Format_RGB888).rgbSwapped()   
        self.scene.addPixmap(QPixmap.fromImage(qImg))
        self.view.setScene(self.scene) 

    def closeEvent(self, event):
        # Prevent the default close behavior
        event.ignore()
        # Hide the window instead
        self.hide()
