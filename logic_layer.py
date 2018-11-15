#!/usr/bin/env python3
# coding: utf-8

import yaml
import cv2
import os
import time
import datetime
import numpy as np
import imagedistort
import queue
from camera import Camera

MODEL_NAME = "one"
MODEL_AUX = "aux"

TIME_DELTA = 1
NUM_PHOTOS = 10
CMD_MARKER = '#'

DISTORT_NUM = 3
FACE_SIZE = 60
IMSHOW_WIDTH = 400

CASCADES_PATH = 'cascades'
FACE_CASCADES = ['lbpcascade_frontalface.xml',
                 'lbpcascade_frontalface_improved.xml',
                 'haarcascade_frontalface_default.xml',
                 'haarcascade_frontalface_alt.xml',
                 'haarcascade_frontalface_alt2.xml']
EYE_CASCADES = ['haarcascade_eye.xml',
                'haarcascade_eye_tree_eyeglasses.xml']

CAM_NUM = 2 
                
cam = Camera(cam_num=CAM_NUM)

def draw_rectangle(img, rect, text=None):
    if rect is not None:
        (x, y, w, h) = rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if text is not None:
            draw_text(img, text, rect[0], rect[1] - 5)
    else:
        center = img.shape[1]//2, img.shape[0]//2
        radius = min(img.shape[:2])//3
        cv2.circle(img, center, radius, (0, 0, 255), 5)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        # 'cascades/haarcascade_frontalface_alt.xml')
        'cascades/lbpcascade_frontalface_improved.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if len(faces) == 0:
        return None, None
    elif len(faces) == 1:
        (x, y, w, h) = faces[0]
    else:
        (x, y, w, h) = max(faces, key=lambda x: x[2] * x[3])
    return gray[y:y+w, x:x+h], (x, y, w, h)

# def detect_face(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # Перебираем разные каскады, пока один из них не сработает
#     for face_casc_name in FACE_CASCADES:
#         face_cascade = cv2.CascadeClassifier(CASCADES_PATH +
#                                              "/" + face_casc_name)
#         faces = face_cascade.detectMultiScale(gray,
#                                               scaleFactor=1.2, minNeighbors=5)
#         # Этот каскад не сработал - берём следующий
#         if len(faces) == 0:
#             continue
#         # Выбираем лицо с максимальной площадью
#         x, y, w, h = max(faces, key=lambda x: x[2] * x[3])
#         # Вырезаем лицо
#         img_face = gray[y:y+h, x:x+w]
#         return img_face, (x, y, w, h)
#         # # Перебираем каскады поиска глаз
#         # for eye_casc_name in EYE_CASCADES:
#         #     eye_cascade = cv2.CascadeClassifier(CASCADES_PATH +
#         #                                         "/" + eye_casc_name)
#         #     eyes = eye_cascade.detectMultiScale(img_face)
#         #     # Нашли правильное лицо, если на нём есть два глаза
#         #     if len(eyes) == 2:
#         #         return img_face, (x, y, w, h)
#     # Ничего на нашлось
#     return None, None


def resize(img, width):
    scl = width / img.shape[1]
    return cv2.resize(img, (0, 0), fx=scl, fy=scl), scl


def equalize(faces, size):
    if size is None:
        # size = 0
        # for face in faces:
        #     size += face.shape[0]
        # size //= len(faces)
        size = 100000
        for face in faces:
            size = min(size, face.shape[0])
    faces_new = []
    for face in faces:
        faces_new.append(cv2.resize(face, (size, size)))
    return faces_new, size


def predict_testing_data(frame_queue, comand_queue, names, size, face_recognizer):
    while comand_queue.empty():
        test_img = cam.get_frame()
        predicted_img, info = predict(test_img, names, size, face_recognizer)
        #print(info)
        frame_queue.put(predicted_img)


def predict(test_img, names, size, face_recognizer):
    img = test_img.copy()
    face, rect = detect_face(img)
    if face is None:
        return img, "No face found"
    face = cv2.resize(face, (size, size))
    label, confidence = face_recognizer.predict(face)
    #info = names[label] + " - {:.3f}".format(confidence)
    info = names[label]
    draw_rectangle(img, rect, info)
    return img, info


def write(img, path, name):
    """ Записать картинку в файл с уникальным именем """
    # Из текущей даты и времени с учётом микросекунд строим уникальный код
    uniq = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    # Получаем имя файла
    file_name = path + "/" + name + "-" + uniq + ".jpg"
    print(file_name)
    return cv2.imwrite(file_name, img, params=[cv2.IMWRITE_JPEG_QUALITY, 80])


class TimeDelta:
    def __init__(self, delta):
        self.delta = delta
        self.crn_time = time.time()

    def __call__(self):
        if self.delta == 0:
            return True
        new_time = time.time()
        if new_time - self.crn_time > self.delta:
            self.crn_time = new_time
            return True
        else:
            return False


def start_photo(frame_queue, name, faces, labels, names):
    try:
        label = names.index(names)
    except ValueError:
        label = len(names)
    names.append(name)
    td = TimeDelta(TIME_DELTA)
    distort = imagedistort.ImageDistort(scale=0.01)
    cnt = 0
    while cnt < NUM_PHOTOS:
        image_orig = cam.get_frame()
        # try:
        #     frame_queue.put_nowait(frm)
        # except queue.Full:
        #     pass

        if td():
            bad_photo = True
            for itr in range(DISTORT_NUM + 1):
                image = image_orig.copy() if itr == 0 else distort(image_orig.copy())
                face, rect = detect_face(image)
                draw_rectangle(image, rect, "Made {} photos".format(cnt))
                try:
                    frame_queue.put_nowait(image)
                except queue.Full:
                    pass
                if face is not None:
                    faces.append(face)
                    labels.append(label)
                    bad_photo = False
            if not bad_photo:
                cnt += 1
    return faces, labels, names

# photo = 1 train = 2 recog = 3
def just_feed(frame_queue, command_queue):
    while command_queue.empty():
        #print("just feed")
        try:
            frame_queue.put_nowait(cam.get_frame())
        except queue.Full:
            pass



def main(frame_queue=None, command_queue=None):
    #print("logic layer")
    faces = []
    labels = []
    names = []
    # File names
    aux_file = MODEL_NAME + "_" + MODEL_AUX + ".yaml"
    model_file = MODEL_NAME + ".yaml"
    retrain = False
    while True:
        # face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer = cv2.face.EigenFaceRecognizer_create()
        # face_recognizer = cv2.face.FisherFaceRecognizer_create()
        if not os.path.isfile(model_file) or retrain:
            while True:
                just_feed(frame_queue, command_queue)
                cmd = command_queue.get()
                if cmd[0] == CMD_MARKER:
                    break
                faces, labels, names = start_photo(frame_queue, cmd, faces, labels, names)
            faces, _ = equalize(faces, FACE_SIZE)
            print("train")
            # Train and save updates model
            face_recognizer.train(faces, np.array(labels))
            with open(aux_file, mode='w') as file:
                yaml.dump((faces, labels, names), file, default_flow_style=False, allow_unicode=True)
            face_recognizer.save(model_file)
        else:
            with open(aux_file, mode='r') as file:
                faces, labels, names = yaml.load(file)
            face_recognizer.read(model_file)
        predict_testing_data(frame_queue, command_queue, names, FACE_SIZE, face_recognizer)
        print("remove")
        retrain = True
        del face_recognizer


if __name__ == '__main__':
    main()
