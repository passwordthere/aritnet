import sys
import cv2
from demo_rotation import main,draw_plot

lookup = {'0': 2,
          '3': 3,
          '6': 4,
          '1': 1,
          '4': 0,
          '7': 5,
          '2': 8,
          '5': 7,
          '8': 6}
result = {
    'rotation_left': ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
    'rotation_right': ['-', '-', '-', '-', '-', '-', '-', '-', '-'],
}
images_list_left = [None, None, None, None, None, None, None, None, None]
images_list_right = [None, None, None, None, None, None, None, None, None]
str_pos_left = sys.argv[1]
str_pos_right = sys.argv[2]
str_path = sys.argv[3]

# str_pos_left = "15"
# str_pos_right = "15"
# str_path = "/home/baiyi/PIC_TEST/安/rotation/"

try:
    for i in range(9):
        images_list_left[lookup[str(i)]] = cv2.imread((str_path + "left_{}.jpg").format(i + 1), cv2.IMREAD_COLOR)

    result['rotation_left'], plot = main(images_list_left, od=False)
    str_l = "1eft"
    result_l = result['rotation_left']
    pic_l = str_path + "plot_left.png"
    draw_plot(result_l, pic_l)
    #cv2.imwrite(str_path, plot)
except Exception:
    str_l = "lefterror"
    result_l = "lefterror"
    pic_l = "lefterror"

try:
    for i in range(9):
        images_list_right[lookup[str(i)]] = cv2.imread((str_path + "right_{}.jpg").format(i + 1), cv2.IMREAD_COLOR)

    result['rotation_right'], plot = main(images_list_right, od=True)
    str_r = "right"
    result_r = result['rotation_right']
    pic_r = str_path + "plot_right.png"
    draw_plot(result_r, pic_r)
    #cv2.imwrite(str_path, plot)
except Exception:
    str_r = "righterror"
    result_r = "righterror"
    pic_r = "righterror"

print(str_l)
print(result_l)
print(pic_l)
print(str_r)
print(result_r)
print(pic_r)