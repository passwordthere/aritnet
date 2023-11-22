import configparser

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import draw

config = configparser.ConfigParser()
config.read("/home/baiyi/soft/OphthalmoscopeSoft/Admin/syslog/Algorithm.ini")
# config.read("Algorithm.ini")

center_position_L = config.getfloat('proptosis_left', 'center_position')
depth_L = config.getfloat('proptosis_left', 'depth')
distance_L = config.getfloat('proptosis_left', 'distance')
focal_length_L = config.getfloat('proptosis_left', 'focal_length')
# k_L = config.getfloat('proptosis_left','k')
mirror_offset_L = config.getfloat('proptosis_left', 'mirror_offset')
degree_L = config.getfloat('proptosis_left', 'degree')
mmPerPixel_L = config.getfloat('width_redness', 'pixelPerMm_L')
pixelPerMm_L = 1 / mmPerPixel_L

center_position_R = config.getfloat('proptosis_right', 'center_position')
depth_R = config.getfloat('proptosis_right', 'depth')
distance_R = config.getfloat('proptosis_right', 'distance')
focal_length_R = config.getfloat('proptosis_right', 'focal_length')
# k_R = config.getfloat('proptosis_right','k')
mirror_offset_R = config.getfloat('proptosis_right', 'mirror_offset')
degree_R = config.getfloat('proptosis_right', 'degree')
mmPerPixel_R = config.getfloat('width_redness', 'pixelPerMm_R')
pixelPerMm_R = 1 / mmPerPixel_R

k_L = (768 - distance_L) / center_position_L
k_R = (768 - distance_R) / center_position_R


def plt_show(seg_map, cv2plt=False):
    if cv2plt:
        seg_map = cv2.cvtColor(seg_map, cv2.COLOR_BGR2RGB)
    plt.imshow(seg_map)
    plt.show()


# def metric2pixel(offset, od):
#     if od:
#         return int(21.6956 * offset + 310)
#     else:
#         return int(21.7982 * offset + (1536 - 1213))


def metric2pixel(offset, od):
    if od:
        return int(k_R * offset + distance_R)
    else:
        return int(-k_L * offset + (1536 - distance_L))


class ElliFit():
    def __init__(self, **kwargs):
        self.data = np.array([])  # Nx2
        self.W = np.array([])
        self.Phi = []
        self.pts_lim = 6 * 2
        for k, v in kwargs.items():
            setattr(self, k, v)
        if np.size(self.W):
            self.weighted = True
        else:
            self.weighted = False
        if np.size(self.data) > self.pts_lim:
            self.model = self.fit()
            self.error = np.mean(self.fit_error(self.data))
        else:
            self.model = [-1, -1, -1, -1, -1]
            self.Phi = [-1, -1, -1, -1, -1]
            self.error = np.inf

    def fit(self):
        # Code implemented from the paper ElliFit
        xm = np.mean(self.data[:, 0])
        ym = np.mean(self.data[:, 1])
        x = self.data[:, 0] - xm
        y = self.data[:, 1] - ym
        X = np.stack([x ** 2, 2 * x * y, -2 * x, -2 * y, -np.ones((np.size(x),))], axis=1)
        Y = -y ** 2
        if self.weighted:
            self.Phi = np.linalg.inv(
                X.T.dot(np.diag(self.W)).dot(X)
            ).dot(
                X.T.dot(np.diag(self.W)).dot(Y)
            )
        else:
            try:
                self.Phi = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))
            except:
                self.Phi = -1 * np.ones(5, )
        try:
            x0 = (self.Phi[2] - self.Phi[3] * self.Phi[1]) / ((self.Phi[0]) - (self.Phi[1]) ** 2)
            y0 = (self.Phi[0] * self.Phi[3] - self.Phi[2] * self.Phi[1]) / ((self.Phi[0]) - (self.Phi[1]) ** 2)
            term2 = np.sqrt(((1 - self.Phi[0]) ** 2 + 4 * (self.Phi[1]) ** 2))
            term3 = (self.Phi[4] + (y0) ** 2 + (x0 ** 2) * self.Phi[0] + 2 * self.Phi[1])
            term1 = 1 + self.Phi[0]
            b = (np.sqrt(abs(2 * term3 / (term1 + term2))))
            a = (np.sqrt(abs(2 * term3 / (term1 - term2))))
            alpha = 0.5 * np.arctan2(2 * self.Phi[1], 1 - self.Phi[0])
            model = [x0 + xm, y0 + ym, a, b, -alpha]
        except:
            print('Inappropriate model generated')
            model = [np.nan, np.nan, np.nan, np.nan, np.nan]
        if np.all(np.isreal(model)) and np.all(~np.isnan(model)) and np.all(~np.isinf(model)):
            model = model
        else:
            model = [-1, -1, -1, -1, -1]
        return model

    def fit_error(self, data):
        # General purpose function to find the residual
        # model: xc, yc, a, b, theta
        term1 = (data[:, 0] - self.model[0]) * np.cos(self.model[-1])
        term2 = (data[:, 1] - self.model[1]) * np.sin(self.model[-1])
        term3 = (data[:, 0] - self.model[0]) * np.sin(self.model[-1])
        term4 = (data[:, 1] - self.model[1]) * np.cos(self.model[-1])
        res = (1 / self.model[2] ** 2) * (term1 - term2) ** 2 + \
              (1 / self.model[3] ** 2) * (term3 + term4) ** 2 - 1
        return np.abs(res)


class ransac():
    def __init__(self, data, model, n_min, mxIter, Thres, n_good):
        self.data = data
        self.num_pts = data.shape[0]
        self.model = model
        self.n_min = n_min
        self.D = n_good if n_min < n_good else n_min
        self.K = mxIter
        self.T = Thres
        self.bestModel = self.model(**{'data': data})  # Fit function all data points

    def loop(self):
        i = 0
        if self.num_pts > self.n_min:
            while i <= self.K:
                # Pick n_min points at random from dataset
                inlr = np.random.choice(self.num_pts, self.n_min, replace=False)
                loc_inlr = np.in1d(np.arange(0, self.num_pts), inlr)
                outlr = np.where(~loc_inlr)[0]
                potModel = self.model(**{'data': self.data[loc_inlr, :]})
                listErr = potModel.fit_error(self.data[~loc_inlr, :])
                inlr_num = np.size(inlr) + np.sum(listErr < self.T)
                if inlr_num > self.D:
                    pot_inlr = np.concatenate([inlr, outlr[listErr < self.T]], axis=0)
                    loc_pot_inlr = np.in1d(np.arange(0, self.num_pts), pot_inlr)
                    betterModel = self.model(**{'data': self.data[loc_pot_inlr, :]})
                    if betterModel.error < self.bestModel.error:
                        self.bestModel = betterModel
                i += 1
        else:
            # If the num_pts <= n_min, directly return the model
            self.bestModel = self.model(**{'data': self.data})
        return self.bestModel


def getValidPoints(seg_map, isPartSeg=True):
    '''
    RK: This can only be used specifically for PartSeg
    Given labels, identify pupil and iris points.
    pupil: label == 3, iris: label ==2
    '''
    im = np.uint8(255 * seg_map.astype(np.float32) / seg_map.max())
    edges = cv2.Canny(im, 50, 100) + cv2.Canny(255 - im, 50, 100)
    r, c = np.where(edges)
    pupilPts = []
    irisPts = []
    for loc in zip(c, r):
        temp = seg_map[loc[1] - 1:loc[1] + 2, loc[0] - 1:loc[0] + 2]
        # condPupil = np.any(temp == 0) or np.any(temp == 1)
        condPupil = ((np.any(temp == 0) and np.any(temp == 3)) or
                     (np.any(temp == 2) and np.any(temp == 3)))
        if isPartSeg:
            condIris = np.any(temp == 0) or np.any(temp == 3) or temp.size == 0
        else:
            condIris = np.any(temp == 3) or temp.size == 0
        # pupilPts.append(np.array(loc)) if not condPupil else None
        pupilPts.append(np.array(loc)) if condPupil else None
        irisPts.append(np.array(loc)) if not condIris else None
    pupilPts = np.stack(pupilPts, axis=0) if len(pupilPts) > 0 else []
    irisPts = np.stack(irisPts, axis=0) if len(irisPts) > 0 else []
    return pupilPts, irisPts


def get_ellipse(pupilPts, seg_map):
    if np.sum(seg_map == 3) > 50 and type(pupilPts) is not list:
        model_pupil = ransac(pupilPts, ElliFit, 15, 40, 5e-3, 15).loop()
        pupil_ellipse = np.array(model_pupil.model)
        return pupil_ellipse


def plot_segmap_ellpreds(image, ellipse, color=np.array([0, 0, 255])):
    if not np.all(ellipse == -1):
        [rr_i, cc_i] = draw.ellipse_perimeter(int(ellipse[1]),
                                              int(ellipse[0]),
                                              int(ellipse[3]),
                                              int(ellipse[2]),
                                              orientation=ellipse[4])
        rr_i = rr_i.clip(6, image.shape[0] - 6)
        cc_i = cc_i.clip(6, image.shape[1] - 6)
        image[rr_i, cc_i, ...] = color


from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        print(f'Function {func.__name__} Took {total_time:.4f} seconds', file=open('log/logs.log', 'a'))
        return result

    return timeit_wrapper
