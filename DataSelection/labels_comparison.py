# Written by Stefano Donne
# stefanodonne@gmail.com
# November 2021

import numpy as np
import cv2 as cv
import pandas as pd
import os
import random
import sys
from sklearn.metrics import cohen_kappa_score, accuracy_score
import selection_tool as st

INPUT_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]


def stat(df):
    y1 = df["corrected site"]
    y2 = df["selected site"]
    print("\n cohen's kappa = " + str(cohen_kappa_score(y1, y2)))
    print("\n % of agreement = " + str(accuracy_score(y1, y2)))


def display(image_path, c):
    """
    display and manage user key input to determine anatomical site
    :param c:
    :param image_path: path of the source image
    :return: true site label
    """
    image = cv.imread(image_path)
    cv.imshow('randomly picked image',
              st.text_on_frame(cv.resize(image, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC), c))
    while True:
        k = cv.waitKey(2) & 0xFF
        if k == ord("p") and c > 0:
            return 'return'
        if k == ord("q"):
            return 'quit'
        pressed, temp_type = st.KeySwitch().key_manager(k)
        if pressed:
            return temp_type


class LabelManager:
    buffer_s = np.array([])
    buffer_v = np.array([])
    buffer_f = np.array([])
    counter = 0
    return_flag = False
    columns_names = ["video", "file_name", "selected site", "corrected site"]
    result = pd.DataFrame(columns=columns_names)
    close_flag = False

    def iterate(self):
        self.image_picker()
        corrected_site = display(INPUT_PATH + "\\" + self.buffer_s[self.counter] +
                                 "\\" + self.buffer_v[self.counter] + "\\" + self.buffer_f[self.counter], self.counter)
        if corrected_site == 'return':
            self.return_back()
        elif corrected_site == 'quit':
            cv.destroyWindow('randomly picked image')
            self.close_flag = True
        else:
            self.result = self.result.append(pd.Series([self.buffer_v[self.counter],
                                                        self.buffer_f[self.counter], self.buffer_s[self.counter],
                                                        corrected_site],
                                                       index=self.columns_names), ignore_index=True)
            self.counter += 1

    def check_picker(self, s, v, f):
        """
        check if the selected image hasn't been selected yet
        :param s: site
        :param v: video (number)
        :param f: file_name
        """
        while ((self.result["video"] == v) & (self.result["file_name"] == f) & (
                self.result["selected site"] == s)).any():
            self.image_picker()

    def array_append(self, s, v, f):
        self.buffer_s = np.append(self.buffer_s, s)
        self.buffer_v = np.append(self.buffer_v, v)
        self.buffer_f = np.append(self.buffer_f, f)

    def array_rm(self):
        self.buffer_s = self.buffer_s[:-1]
        self.buffer_v = self.buffer_v[:-1]
        self.buffer_f = self.buffer_f[:-1]

    def image_picker(self):
        """
        :return: a tuple containing
        picked_site : which anatomical site has been randomly selected
        picked_video : which video has been randomly selected
        picked_frame : which frame from the video and the given anatomical site has been randomly selected
        """
        if self.return_flag:
            self.return_flag = False
            return 0
        picked_site = random.choice(os.listdir(INPUT_PATH))
        while not os.listdir(INPUT_PATH + "\\" + picked_site):
            picked_site = random.choice(os.listdir(INPUT_PATH))
        picked_video = random.choice(os.listdir(INPUT_PATH + "\\" + picked_site))
        while len(os.listdir(INPUT_PATH + "\\" + picked_site + "\\" + picked_video)) == 0:
            picked_video = random.choice(os.listdir(INPUT_PATH + "\\" + picked_site))
        picked_frame = random.choice(os.listdir(INPUT_PATH + "\\" + picked_site + "\\" + picked_video))

        self.array_append(picked_site, picked_video, picked_frame)

    def save(self):
        stat(self.result)
        self.result.to_excel(OUTPUT_PATH)
        os.startfile(OUTPUT_PATH)

    def return_back(self):
        self.result.drop(self.result.tail(1).index, inplace=True)
        self.array_rm()
        self.counter -= 1
        self.return_flag = True


if __name__ == '__main__':
    lm = LabelManager()
    while not lm.close_flag:
        lm.iterate()
    lm.save()
