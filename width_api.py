import sys
import cv2
from demo_width import width
from demo_utils import *

flow_w = 768
flow_h = 512
ratio = flow_h / 1024

def get_width_result(frame,strpath,right):
    filetxt = open(strpath, "r")
    listOfLines = filetxt.readlines()
    filetxt.close()
    point_x = np.array([])
    point_y = np.array([])
    for line in listOfLines:
        str = line.strip().split(',')
        point_x = np.append(point_x, int(str[0]))
        point_y = np.append(point_y, int(str[1]))
    # # point_x = np.array([])
    # # point_y = np.array([])
    # point_x, point_y = txt_numpyarray(strpath)
    point_up_x = int(point_x[0])
    point_down_x = int(point_x[1])
    point_up_y = int(point_y[0])
    point_down_y = int(point_y[1])
    if right:
        eye_width = abs(round((point_down_y - point_up_y) * mmPerPixel_R, 1))
        frame = cv2.line(frame, (point_up_x, point_up_y), (point_down_x, point_down_y), color=(0, 255, 0), thickness=1)
    else:
        eye_width = abs(round((point_down_y - point_up_y) * mmPerPixel_L, 1))
        frame = cv2.line(frame, (point_up_x, point_up_y), (point_down_x, point_down_y), color=(0, 255, 0), thickness=1)
    return eye_width, frame

str_direction = sys.argv[1]
str_pos = sys.argv[2]
str_pathpre = sys.argv[3]
str_path = sys.argv[4]
str_path_file = sys.argv[5]
str_auto = sys.argv[6]
str_path_show = str_path.replace(".jpg", "_show.jpg")

# str_direction = "left"
# str_pos = "12"
# str_pathpre = "/home/baiyi/PIC_TEST/安/left.jpg"
# str_path = "/home/baiyi/PIC_TEST/安/right_11.jpg"
# str_path_file = "/home/baiyi/PIC_TEST/安/"
# str_auto = "Auto"
# str_path_show = str_path.replace(".jpg", "_show.jpg")

img = cv2.imread(str_pathpre)

try:
    if str_direction == "left":
        if str_auto == "Auto":
            frame_O, result, sig = width(img, float(str_pos), od=False)
        else:
            str_path_txt = str_path_file + "left_width_txt.txt"
            result, frame_O = get_width_result(img, str_path_txt, False)
            sig = frame_O
        print("left")
    else:
        if str_auto == "Auto":
            frame_O, result, sig = width(img, float(str_pos), od=True)
        else:
            str_path_txt = str_path_file + "right_width_txt.txt"
            result, frame_O = get_width_result(img, str_path_txt, True)
            sig = frame_O
        print("right")
    cv2.imwrite(str_path, frame_O)
    cv2.imwrite(str_path_show, sig)
    print("%.1f" % result)
    print(str_path)
    print(str_path_show)
except Exception:
    print(str_direction + "error")
    print(str_direction + "error")
    print(str_direction + "error")
    print(str_direction + "error")