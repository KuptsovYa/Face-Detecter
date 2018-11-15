#!/usr/bin/env python3
# coding: utf-8


import cv2
import os
import numpy as np
import yaml


MODEL_NAME = "one"
MODEL_AUX = "aux"

TRAIN_DATA = "training-data"
TEST_DATA = "test-data"
ALREADY_FACES = False

DISTORT_NUM = 100

IMSHOW_WIDTH = 400

CASCADES_PATH = 'cascades'
FACE_CASCADES = ['lbpcascade_frontalface.xml',
                 'lbpcascade_frontalface_improved.xml',
                 'haarcascade_frontalface_default.xml',
                 'haarcascade_frontalface_alt.xml',
                 'haarcascade_frontalface_alt2.xml']
EYE_CASCADES = ['haarcascade_eye.xml',
                'haarcascade_eye_tree_eyeglasses.xml']


class ImageDistort(yaml.YAMLObject):
    """Предполагаем, что рисунок сюда попадает сразу после imread.
    Тип пикселей - uint, 0-255."""
    yaml_tag = u'!ImageDistort'

    def __init__(self, scale=0.05, max_ops=None):
        # self.flip,
        self.ops = [self.rotate, self.affine, self.perspec, self.noise,
                    self.blur, self.contrast, self.brightness, self.unsharp]
        self.setup(scale, max_ops)

    def __repr__(self):
        return "{cls}(scale={scale!r},max_ops={max_ops!r})".format(
                cls=self.__class__.__name__,
                scale=self.scale, max_ops=self.max_ops)

    def setup(self, scale, max_ops=None):
        self.scale = scale
        self.max_ops = max_ops
        self.max_max_ops = len(self.ops) if max_ops is None else max_ops
        self.scl_rotate = scale
        self.scl_affine = scale
        self.scl_perspec = scale
        self.scl_noise = scale
        self.scl_bright = 10 * scale

    def __call__(self, im):
        # сколько преобразований будет сделано. Возможные значение
        # от 0 до self.max_ops
        nops = np.random.randint(self.max_max_ops+1)

        # Функция np.random.choises делает случайную выборку из
        # набор, заданного своим первым аргументом. Параметр
        # replace разрешает выбирать одно и тоже значение несколько
        # раз. Если запланировано выполнить больше операций чем есть,
        # то нужно повторять некоторые операции по несколько раз.
        # Для этого нужно задать параметр replace=True. А если
        # операций мало, то интересно сделать как можно больше
        # разных операций. Тогда replace=False
        if nops <= len(self.ops):
            replace = False
        else:
            replace = True
        opseq = np.random.choice(self.ops, nops, replace=replace)
        # выполнение преобразваний
        for op in opseq:
            im = op(im)
        return im

    def rotate(self, im):
        deg_max = 90  # маскимальный угол, соответствующий scl_rotate=1
        deg = self.scl_rotate * (2 * np.random.random_sample() - 1) * deg_max
        rows, cols = im.shape[:2]
        M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0),
                                    deg, 1)
        return cv2.warpAffine(im, M, (cols, rows))

    def affine(self, im):
        # размеры рисунка
        shape = im.shape[:2]
        # задаём три реперные в относительных координатах
        rel_0 = np.array([[0.3, 0.5], [0.7, 0.3], [0.7, 0.7]])
        # ищем исходные и возмущёные точки
        pts_0, pts_1 = self.__pert_points(rel_0, shape, self.scl_affine)
        # строим матрицу преобразования и выполяем само преобразование
        mat = cv2.getAffineTransform(pts_0, pts_1)
        return cv2.warpAffine(im, mat, (shape[1], shape[0]))

    def perspec(self, im):
        # размеры рисунка
        shape = im.shape[:2]
        # задаём четыре реперные в относительных координатах
        rel_0 = np.array([[0.2, 0.2], [0.2, 0.8], [0.8, 0.8], [0.8, 0.2]])
        # ищем исходные и возмущёные точки
        pts_0, pts_1 = self.__pert_points(rel_0, shape, self.scl_perspec)
        # строим матрицу преобразования и выполяем само преобразование
        mat = cv2.getPerspectiveTransform(pts_0, pts_1)
        return cv2.warpPerspective(im, mat, (shape[1], shape[0]))

    def noise(self, im):
        mx = im.max()
        tp = im.dtype
        pert = mx * self.scl_noise * (2 * np.random.random_sample(im.shape)-1)
        im1 = cv2.normalize(im.astype('float')+pert, None, 0, mx,
                            norm_type=cv2.NORM_MINMAX)
        return im1.astype(tp)

    def flip(self, im):
        return cv2.flip(im, 1)

    def blur(self, im):
        return cv2.blur(im, (5, 5))

    def contrast(self, im):
        l, a, b = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2LAB))
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    def brightness(self, im):
        value = int(255 * self.scl_bright * np.random.rand())
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if np.random.randint(2) == 0:
            # Увеличить яркость
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            # Уменьшить яркость
            lim = value
            v[v < lim] = 0
            v[v >= lim] -= value
        final_hsv = cv2.merge((h, s, v))
        im = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return im

    def unsharp(self, im):
        gauss = cv2.GaussianBlur(im, (9, 9), 10.0)
        return cv2.addWeighted(im, 1.5, gauss, -0.5, 0, im)

    def __pert_scale(self, scl):
        return scl * (2 * np.random.rand(2) - 1) + np.array([1.0, 1.0])

    def __pert_points(self, rel_0, shape, scale):
        # Для заданных в относительных координатах точке rel_0 получить
        # cлучайное возмущение и перевести в физические координаты

        # rel_0 - точки в относительных координатах
        # shape=[rows,cols]
        # scale - масштаб

        # задём возмущение
        rel_1 = []
        for pt in rel_0:
            rel_1.append(pt * self.__pert_scale(scale))
        rel_1 = np.array(rel_1)
        # переходим в физические координаты
        pts_0, pts_1 = [], []
        for p0, p1 in zip(rel_0, rel_1):
            pts_0.append(p0 * shape)
            pts_1.append(p1 * shape)
        return np.float32(pts_0), np.float32(pts_1)


def draw_rectangle(img, rect):
    if rect is not None:
        (x, y, w, h) = rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
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
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.2, minNeighbors=5)
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


def prepare_training_data(path, old_files, old_labels, old_names):
    # Lst to hold all subject faces
    faces = []
    # Corresponding filenames
    files = []
    # List to hold labels for all subjects
    labels = []
    # Index of name in this list equals to its label
    names = []
    # Total number of faces
    orig_count, total_count, found_count = 0, 0, 0
    # Old names go first, then new ones sorted
    new_names = set(os.listdir(path)) - set(old_names)
    names = old_names + sorted(list(new_names))
    # Image distorsion
    distort = ImageDistort(scale=0.01)
    # Iterate through each directory and read images within it
    for label, dir_name in enumerate(names):
        subject_dir_path = path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        # Exclude images already processed
        old_images_names = [ofl for ofl, lab in zip(old_files, old_labels)
                            if lab == label]
        new_images_names = sorted(list(set(subject_images_names) -
                                       set(old_images_names)))
        # Iterate over new images
        for image_name in new_images_names:
            # Ignore system files
            if image_name.startswith("."):
                continue
            bad_photo = True
            image_path = subject_dir_path + "/" + image_name
            image_orig = cv2.imread(image_path)
            orig_count += 1
            for itr in range(DISTORT_NUM + 1):
                total_count += 1
                image = image_orig.copy() if itr == 0 \
                    else distort(image_orig.copy())
                if itr == 0 and ALREADY_FACES:
                    face = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    face, rect = detect_face(image)
                    draw_rectangle(image, rect)
                if face is not None:
                    faces.append(face)
                    files.append(image_name)
                    labels.append(label)
                    found_count += 1
                    bad_photo = False
                image, _ = resize(image, IMSHOW_WIDTH)
                cv2.imshow("Will train on image...", image)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    raise KeyboardInterrupt
            if bad_photo:
                print("No faces found in {}".format(image_path))
    cv2.destroyAllWindows()

    # Print total faces and found names
    print("Orig  images: ", orig_count)
    print("Total images: ", total_count)
    print("Found faces:  ", found_count)
    print("Names found:  ", names)

    return faces, files, labels, names


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


def predict(test_img, names, size, face_recognizer):
    img = test_img.copy()
    face, rect = detect_face(img)
    if face is None:
        return img, "No face found"
    face = cv2.resize(face, (size, size))
    label, confidence = face_recognizer.predict(face)
    info = names[label] + " - {:.3f}".format(confidence)
    img, scl = resize(img, IMSHOW_WIDTH)
    rect = list(map(lambda x: int(scl * x), rect))
    draw_rectangle(img, rect)
    draw_text(img, info, rect[0], rect[1]-5)
    return img, info


def predict_testing_data(path, names, size, face_recognizer):
    for num, file_name in enumerate(os.listdir(path)):
        pth = path + "/" + file_name
        test_img = cv2.imread(pth)
        predicted_img, info = predict(test_img, names, size, face_recognizer)
        cv2.imshow("{}. {}".format(num, info), predicted_img)
        print("{:>30}: {}".format(pth, info))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # File names
    aux_file = MODEL_NAME + "_" + MODEL_AUX + ".yaml"
    model_file = MODEL_NAME + ".yaml"
    # Read previous aux data
    if os.path.isfile(aux_file):
        with open(aux_file, mode='r') as file:
            old_files, old_labels, old_names, size = yaml.load(file)
    else:
        old_files, old_labels, old_names, size = [], [], [], 60
    # Read directory
    print("Preparing data...")
    faces, files, labels, names = prepare_training_data(TRAIN_DATA, old_files,
                                                        old_labels, old_names)
    faces, size = equalize(faces, size)
    print("Data prepared")

    # Create our LBPH face recognizer
    # face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # or use EigenFaceRecognizer by replacing above line with
    face_recognizer = cv2.face.EigenFaceRecognizer_create()

    # or use FisherFaceRecognizer by replacing above line with
    # face_recognizer = cv2.face.FisherFaceRecognizer_create()

    # Load previous data if exist
    if os.path.isfile(model_file):
        face_recognizer.read(model_file)

    # Train and save updates model
    if len(faces) > 0:
        face_recognizer.train(faces, np.array(labels))
        face_recognizer.save(model_file)
        # Remove duplicated filenames due to distorsion
        files_labels = list(set(list(zip(files, labels))))
        files, labels = [list(x) for x in zip(*files_labels)]
        # Save new aux data
        files = old_files + files
        labels = old_labels + labels
        with open(aux_file, mode='w') as file:
            yaml.dump((files, labels, names, size), file,
                      default_flow_style=False, allow_unicode=True)

    print("Predicting images")
    predict_testing_data(TEST_DATA, names, size, face_recognizer)


if __name__ == '__main__':
    main()
