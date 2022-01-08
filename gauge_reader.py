import logging
import argparse
import math
from typing import Tuple

import cv2
import numpy as np
from numpy import linalg as LA

logger = logging.getLogger(__name__)

threshold = 120


def get_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """2点間の距離を算出

    Args:
        x1 (int): 1点目のX座標
        y1 (int): 1点目のY座標
        x2 (int): 2点目のX座標
        y2 (int): 2点目のY座標

    Returns:
        flaat: 距離
    """
    d: float = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return d


def tangent_angle(u: np.ndarray, v: np.ndarray) -> float:
    """2つのベクトルの角度を算出

    Args:
        u (np.ndarray): 1つ目のベクトル
        v (np.ndarray): 2つ目のベクトル

    Returns:
        float: 角度
    """
    i = np.inner(u, v)
    n = LA.norm(u) * LA.norm(v)
    c = i / n
    return np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))


def get_center(img: np.ndarray) -> Tuple[int, int, int, int, Tuple[int, int]]:
    """入力画像から中心を求める

    Args:
        img (np.ndarray): 入力画像

    Returns:
        Tuple[int, int, int, int, Tuple[int, int]]: 計測版(丸)の座標(x, y, w, h),
                                                    中心の座標(center)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to gray

    _, img_thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    if logger.isEnabledFor(logging.DEBUG):
        cv2.imwrite('debug1.jpg', img_thresh)

    contours, _ = cv2.findContours(img_thresh,
                                   cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)

    for i in range(0, len(contours)):
        if len(contours[i]) > 0:
            if cv2.contourArea(contours[i]) < 1000000:
                continue

            cv2.polylines(img, contours[i], True, (255, 255, 255), 5)
            x, y, w, h = cv2.boundingRect(contours[i])
    offset_y = 35
    circle_width1 = 270
    cv2.circle(img,
               (x + int(w / 2), y + int(h / 2) - offset_y),
               circle_width1,
               (255, 255, 255),
               -1,
               cv2.LINE_AA)  # draw circle
    cv2.circle(img,
               (x + int(w / 2), y + int(h / 2) - offset_y),
               int(w / 2) - 110,
               (255, 255, 255),
               100,
               cv2.LINE_AA)  # draw circle
    center = (x + int(w / 2), y + int(h / 2) - offset_y + 10)
    if logger.isEnabledFor(logging.DEBUG):
        cv2.imwrite('debug2.jpg', img)

    return x, y, w, h, center


def get_meter_coordinates(img: np.ndarray,
                          x: int, y: int,
                          w: int, h: int,
                          center: Tuple[int, int]) -> Tuple[int, int]:
    """画像からメーターの先端の座標を取得

    Args:
        img (np.ndarray): 入力画像
        x (int): [description]
        y (int): [description]
        w (int): [description]
        h (int): [description]
        center (Tuple[int, int]): 中心の座標(x, y)

    Returns:
        Tuple[int, int]: メーターの先端の座標(x, y)
    """
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to gray
    _, img_thresh2 = cv2.threshold(gray2, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img_thresh2,
                                   cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)

    for i in range(0, len(contours)):
        if len(contours[i]) > 0:

            # remove small objects
            if cv2.contourArea(contours[i]) < 2600 \
               or cv2.contourArea(contours[i]) > 4000:
                continue

            x_tmp, y_tmp, w_tmp, h_tmp = cv2.boundingRect(contours[i])
            if x_tmp > x + 100 and x_tmp < x + w - 200 \
               and y_tmp > y + 100 and y_tmp < y + h - 200 \
               and w_tmp < 300 and h_tmp < 300:
                cv2.rectangle(img,
                              (x_tmp, y_tmp),
                              (x_tmp + w_tmp, y_tmp + h_tmp),
                              (255, 255, 0))
                x2 = x_tmp
                y2 = y_tmp
                w2 = w_tmp
                h2 = h_tmp

    # 一番中心から遠い点を求める
    vertexs = [(x2, y2), (x2, y2 + h2), (x2 + w2, y2), (x2 + w2, y2 + h2)]
    distance_max = 0
    point = (None, None)
    for vertex in vertexs:
        d = get_distance(center[0], center[1], vertex[0], vertex[1])
        if d > distance_max:
            distance_max = d
            point = vertex
    if logger.isEnabledFor(logging.DEBUG):
        cv2.imwrite('debug3.jpg', img)
    return point


def calc_value(center: Tuple[int, int], point: Tuple[int, int]) -> int:
    """与えられた針の座標から角度を計算し、値として返す

    Args:
        center (Tuple[int, int]): 中心の座標(x, y)
        point (Tuple[int, int]): 針の座標(x, y)

    Returns:
        int: 針が示す計測値(0-1000) 計測不能のときは -1
    """
    # ベクトルに直す
    a = np.array([-266, 432])
    b = np.array([point[0] - center[0], point[1] - center[1]])

    if point[0] < center[0]:
        value = int(tangent_angle(a, b) * (605/180))
    else:
        value = int((360 - tangent_angle(a, b)) * (605/180))
    if 0 <= value <= 1000:
        return value
    else:
        return -1


def main(input):
    img = cv2.imread(input)
    if logger.isEnabledFor(logging.DEBUG):
        img_debug = img.copy()

    try:
        x, y, w, h, center = get_center(img)
        point = get_meter_coordinates(img, x, y, w, h, center)
        value = calc_value(center, point)
        if logger.isEnabledFor(logging.DEBUG):
            cv2.line(img_debug,
                     pt1=center,
                     pt2=point,
                     color=(0, 255, 0),
                     thickness=3,
                     lineType=cv2.LINE_4,
                     shift=0)
            cv2.imwrite('debug4.jpg', img_debug)
    except Exception:
        logger.debug("Return Exception")
        value = -1
    print(f'{value} g')


def parse_args():
    # オプションの解析
    parser = argparse.ArgumentParser(description='OpenPoseの実行')

    parser.add_argument(
                        'input',
                        )
    parser.add_argument(
                        '-l', '--loglevel',
                        choices=('warning', 'debug', 'info'),
                        default='info'
                        )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logger.setLevel(args.loglevel.upper())
    logger.info('loglevel: %s', args.loglevel)
    lformat = '%(name)s <L%(lineno)s> [%(levelname)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=lformat,
    )
    logger.setLevel(args.loglevel.upper())

    main(args.input)
