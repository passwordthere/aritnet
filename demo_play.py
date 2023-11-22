import os
import shutil

import cv2
import numpy as np
import torch

from dataset import transform
from models import model_dict
from utils import get_predictions, Logger

os.makedirs('log', exist_ok=True)
if len(os.listdir('log')) > 1024:
    shutil.rmtree('log')
logger = Logger(os.path.join('log', 'logs.log'))


def segmentation(frame, redness=False):
    model_path = 'flip_discard.pkl'
    if redness:
        model_path = 'redness_baseonbest.pkl'
    model = model_dict['densenet']
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model.eval()

    with torch.no_grad():
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (768, 512))

        table = 255.0 * (np.linspace(0, 1, 256) ** 0.8)
        frame = cv2.LUT(frame, table)

        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        frame = clahe.apply(np.array(np.uint8(frame)))

        data = transform(frame).unsqueeze(0).cuda()
        output = model(data)
        predict = get_predictions(output).cpu().squeeze().numpy()

    return predict


if __name__ == '__main__':
    # od = True
    # for i in Path.cwd().rglob('les_0/{}*.bmp'.format('od' if od else 'os')):
    #     frame = cv2.imread(str(i))
    #     if not od:
    #         frame = cv2.flip(frame, 1)
    #     predict = segmentation(frame)
    #
    #     frame = cv2.resize(frame, (768, 512))
    #     predict = cv2.normalize(src=predict, dst=None, alpha=0, beta=255, norm_ty pe=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #     predict = cv2.applyColorMap(predict, 16)
    #     weights = 0.4
    #     combined = cv2.addWeighted(frame, weights, predict, 1 - weights, 0)
    #
    #     # plt_show(combined, True)
    #     cv2.imwrite('les_0_out/{}.jpg'.format(i.stem), combined)

    frame = cv2.imread('0000.bmp')
    frame = cv2.imread('0-prop-xiaozhuOD.jpg')
    # frame = cv2.flip(frame, 1)
    predict = segmentation(frame)

    frame = cv2.resize(frame, (768, 512))
    predict = cv2.normalize(src=predict, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    predict = cv2.applyColorMap(predict, 16)
    weights = 0.4
    combined = cv2.addWeighted(frame, weights, predict, 1 - weights, 0)

    # plt_show(combined, True)
    cv2.imwrite('0000.jpg', combined)
