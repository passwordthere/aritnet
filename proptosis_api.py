import sys
import cv2
from demo_proptosis import proptosis

str_direction = sys.argv[1]
str_pos = sys.argv[2]
str_pathpre = sys.argv[3]
str_path = sys.argv[4]
str_path_show = str_path.replace(".jpg", "_show.jpg")
img = cv2.imread(str_pathpre)
try:
    if str_direction == "left":
        result, frame_O, sig = proptosis(img, float(str_pos), od=False)
        cv2.imwrite(str_path, frame_O)
        cv2.imwrite(str_path_show, sig)
        print("left")
    else:
        result, frame_O, sig = proptosis(img, float(str_pos), od=True)
        cv2.imwrite(str_path, frame_O)
        cv2.imwrite(str_path_show, sig)
        print("right")
    print("%.1f" % result)
    print(str_path)
    print(str_path_show)
except Exception:
    print(str_direction + "error")
    print(str_direction + "error")
    print(str_direction + "error")
    print(str_direction + "error")
