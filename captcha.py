import cv2 as cv
import time
import argparse
import os
import shutil
import captcha_util as captcha


def get_key_x(item):
    return item[1][0]


def detect_recog(img):
    start_time = time.time()

    _, recog_boxes, _, classids = captcha.infer_image(img)

    if classids is None or len(classids) == 0:
        return ""

    info = []
    for i in range(len(classids)):
        info.append((labels[classids[i]], recog_boxes[i]))

    info = sorted(info, key=get_key_x)

    result = ''
    for item in info:
        result += item[0]

    return result


if __name__ == '__main__':
    labels = "0123456789"
    path = "test/Solved-600"

    file_list = os.listdir(path)
    total = 0
    c_total = 0
    for afile in file_list:
        filename, file_extension = os.path.splitext(afile)
        if file_extension != ".jpg":
            continue

        img_path = path + "/" + filename + ".jpg"
        start_time = time.time()
        img = cv.imread(img_path)
        res = detect_recog(img)

        total += 1
        if filename == res:
            c_total += 1
        # else:
            print(filename + '.jpg', res, time.time()-start_time)
        else:
            print(filename + '.jpg', res, time.time() - start_time, 'failed')

    print(total, c_total, c_total/total*100, '%')


    # parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
    # parser.add_argument('--image', help='Path to image file.')
    # args = parser.parse_args()
    #
    # outputFile = "out_py.mp4"
    #
    # image_path = args.image
    # if (image_path):
    #     start_time = time.time()
    #     img = cv.imread(image_path)
    #     res = detect_recog(img)
    #     print(res)
    #     cv.imshow("test", img)
    #     cv.waitKey(0)

