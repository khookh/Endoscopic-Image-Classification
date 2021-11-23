# Written by Stefano Donne
# stefanodonne@gmail.com
# November 2021

import cv2 as cv
import pandas as pd
import os
import random
import sys
import selection_tool as st

INPUT_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]


def image_picker():
    """
    :return: a tuple containing
    picked_site : which anatomical site has been randomly selected
    picked_video : which video has been randomly selected
    picked_frame : which frame from the video and the given anatomical site has been randomly selected
    """
    picked_site = random.choice(os.listdir(INPUT_PATH))
    while not os.listdir(INPUT_PATH + "\\" + picked_site):
        picked_site = random.choice(os.listdir(INPUT_PATH))
    picked_video = random.choice(os.listdir(INPUT_PATH + "\\" + picked_site))
    picked_frame = random.choice(os.listdir(INPUT_PATH + "\\" + picked_site + "\\" + picked_video))
    return picked_site, picked_video, picked_frame


def display(image_path):
    """
    display and manage user key input to determine anatomical site
    :param image_path: path of the source image
    :return: true site label
    """
    image = cv.imread(image_path)
    while True:
        k = cv.waitKey(2) & 0xFF
        pressed, temp_type = st.KeySwitch().key_manager(k)
        if pressed:
            return temp_type
        cv.imshow('randomly picked image',
                  st.text_on_frame(cv.resize(image, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)))


if __name__ == '__main__':
    counter = 100
    columns_names = ["video", "file_name", "selected site", "corrected site"]
    result = pd.DataFrame(columns=columns_names)
    while counter > 0:
        site, video, image_name = image_picker()
        corrected_site = display(INPUT_PATH + "\\" + site + "\\" + video + "\\" + image_name)
        result = result.append(pd.Series([video, image_name, site, corrected_site], index=columns_names),
                               ignore_index=True)
        counter -= 1
    result.to_excel(OUTPUT_PATH)
