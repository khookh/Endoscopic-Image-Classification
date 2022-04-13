import cv2 as cv
import os

INPUT_PATH = 'E:\\Gastroscopies\\Dataset'
OUTPUT_PATH = 'E:\\Gastroscopies\\DataSet_cropped'


def crop(img):
    """
    Crop the frame for accurate DNN classification
    :param img: to be cropped
    :return: cropped image
    """
    size = img.shape
    if size[1] / size[0] > 1.35:
        h, s, v = cv.split(img)
        x1 = 1
        x2 = size[1] - 1
        y1 = 1
        y2 = size[0] - 1
        while v[int(size[0] / 2), x1] < 15:
            x1 += 5
        while v[int(size[0] / 2), x2] < 15:
            x2 -= 5
        while v[y1, int(size[1] / 2)] < 15:
            y1 += 5
        while v[y2, int(size[1] / 2)] < 15:
            y2 -= 5
        return img[y1:y2, x1:x2]
    else:
        return img


if __name__ == '__main__':
    for site in os.listdir(INPUT_PATH):
        print(site)
        for picture in os.listdir(INPUT_PATH + "\\" + site):
            cv.imwrite(OUTPUT_PATH + "\\" + site + "\\" + picture,
                       crop(cv.imread(INPUT_PATH + "\\" + site + "\\" + picture)))