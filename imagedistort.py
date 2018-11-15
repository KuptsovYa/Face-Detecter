import cv2
import numpy as np
import yaml


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
