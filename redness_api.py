import sys
import cv2
import numpy as np
from demo_redness import redness

def get_pts(str_path_txt):
    with open(str_path_txt,'r') as file:
        lines = file.readlines()
    coordinate_str = []
    for line in lines:
        x, y = line.strip().split(',')
        coordinate_str.append([int(x),int(y)])
    #coordinate_matrix = np.array(coordinate_str)
    return coordinate_str


str_direction = sys.argv[1]
str_pos = sys.argv[2]
str_pathpre = sys.argv[3]
str_path = sys.argv[4]
str_path_file = sys.argv[5]
str_auto = sys.argv[6]
# str_direction = "right"
# str_pos = "12"
# str_pathpre = "/home/baiyi/PIC_TEST/安/redness_od.bmp"
# str_path = "/home/baiyi/PIC_TEST/安/redness_11.bmp"
# str_path_file = "/home/baiyi/PIC_TEST/安/"
# str_auto = "man"
img = cv2.imread(str_pathpre)
#redness_frame, redness_value = redness(img, float(str_pos), od=True)

try:
    if str_direction == "left":
        if str_auto == "Auto":
            redness_frame, redness_value = redness(img, float(str_pos), od=False)
        else:
            str_path_txt = str_path_file + "left_redness_txt.txt"
            point = get_pts(str_path_txt)
            redness_frame, redness_value = redness(img, float(str_pos), od=False, pts=point)
        print("left")
    else:
        if str_auto == "Auto":
            redness_frame, redness_value = redness(img, float(str_pos), od=True)
        else:
            str_path_txt = str_path_file + "right_redness_txt.txt"
            point = get_pts(str_path_txt)
            redness_frame, redness_value = redness(img, float(str_pos), od=True, pts=point)
        print("right")
    cv2.imwrite(str_path, redness_frame)
    print(redness_value)
    print(str_path)
except Exception:
    print(str_direction + "error")
    print(str_direction + "error")
    print(str_direction + "error")




