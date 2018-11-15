# -*- coding: utf-8 -*-

"""Работа с камерой в отдельном процессе."""


import cv2
import multiprocessing as mp
import queue
import time


class CameraError(IOError):
    pass

def identify(x):
    return x

class Camera:
    filter = identify

    def __init__(self, cam_num=0, cam_wh=(320, 240)):
        self.cam_num = cam_num
        self.cam_wh = cam_wh
        self.break_flag = mp.Value('i', 0)  # flag to break proc
        self.state = mp.Value('i', -1)  # state of proc
        self.last_access = mp.Value('d', time.time())
        self.frame = mp.Queue(maxsize=5)  # queue for transferring frames
        # start background frame proc
        self.proc = mp.Process(target=self._process,
                               args=(self.frame, self.break_flag, self.state, self.last_access))
        self.proc.start()
        # wait for successfull initialization
        while self.state.value < 0:
            pass
        # initialization failed
        if self.state.value > 0:
            raise CameraError("Problem {} with camera {}"
                              .format(self.state.value, self.cam_num))

    def __del__(self):
        # print("__del__")
        # Signal to stop _process
        self.break_flag.value = 1
        # Wait until _process actually stops
        self.proc.join()

    def get_frame(self):
        while True:
            if self.proc.is_alive():
                try:
                    self.last_access.value = time.time()
                    frame = self.frame.get_nowait()
                    break
                except queue.Empty:
                    pass
            else:
                raise CameraError("Problem {} with camera {}"
                                  .format(self.state.value, self.cam_num))
        return frame

    def _process(self, frame, break_flag, state, last_access):
        try:
            # Initialize camera
            cam = cv2.VideoCapture(self.cam_num)
            if not cam.isOpened():
                state.value = 1
            else:
                # Size of frame
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_wh[0])
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_wh[1])
                state.value = 0
                while True:
                    # Catch next frame
                    # print("cam.read()")
                    ret, img = cam.read()
                    if not ret:
                        state.value = 2
                        break
                    try:
                        frame.put_nowait(__class__.filter(img))
                    except queue.Full:
                        pass
                    # frame.put(cv2.imencode( '.jpg', img)[1].tobytes())
                    if break_flag.value == 1 or time.time() - last_access.value > 10:
                        break
        except KeyboardInterrupt:
            pass
        finally:
            # print("process finishing, state = ", state.value)
            cam.release()
            frame.close()
            frame.cancel_join_thread()
        return
