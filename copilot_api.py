import sys
import cv2
from demo_copilot import copilot

str_path = sys.argv[1]
#str_path= "/home/baiyi/PIC_TEST/安/"
try:
    frame = cv2.imread(str_path+"right.jpg")
    move_x1, move_y1 = copilot(frame, 0, od=True)
    '''os'''
    frame = cv2.imread(str_path+"left.jpg")
    move_x2, move_y2 = copilot(frame, 0, od=False)
    move_x, move_y = move_x1 + move_x2, move_y1 + move_y2
    print(move_x)
    print(move_y)
except Exception:
    print("error")
    print("error")