"""Камера без создания отдельного процесса."""

import cv2
import numpy as np


class CameraError(IOError):
    pass


class Camera:
    filter = lambda x: x

    def __init__(self, cam_num=0, cam_wh=(320, 240)):
        self.cam_num = cam_num
        self.cam_wh = cam_wh
        self.cam = cv2.VideoCapture(self.cam_num)
        if not self.cam.isOpened():
            raise CameraError("Problem 1 with camera {}".format(self.cam_num))
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_wh[0])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_wh[1])

    def __del__(self):
        self.cam.release()

    def get_frame(self):
        ret, img = self.cam.read()
        if not ret:
            raise CameraError("Problem 2 with camera {}".format(self.cam_num))
        #img = np.hstack([img, __class__.filter(img)])
        return img
