import numpy as np
import cv2 as cv
import sys
import os

from multiprocessing import Queue

INPUT_PATH = sys.argv[1]
DATA_PATH = sys.argv[2]


def textonframe(dispframe_):
    dispframe_ = cv.putText(dispframe_, "0 = oesophagus", (5, 20), cv.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1,
                            cv.LINE_AA)
    dispframe_ = cv.putText(dispframe_, "1 = junction", (5, 40), cv.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1,
                            cv.LINE_AA)
    dispframe_ = cv.putText(dispframe_, "2 = grande courbure gastrique", (5, 60), cv.FONT_HERSHEY_SIMPLEX, .4,
                            (0, 0, 255), 1,
                            cv.LINE_AA)
    dispframe_ = cv.putText(dispframe_, "3 = pylore-antre", (5, 80), cv.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1,
                            cv.LINE_AA)
    dispframe_ = cv.putText(dispframe_, "4 = angle", (5, 100), cv.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1,
                            cv.LINE_AA)
    dispframe_ = cv.putText(dispframe_, "5 = retro vision", (5, 120), cv.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1,
                            cv.LINE_AA)
    dispframe_ = cv.putText(dispframe_, "6 = sur angulaire-petite courbure", (5, 140), cv.FONT_HERSHEY_SIMPLEX, .4,
                            (0, 0, 255), 1,
                            cv.LINE_AA)
    dispframe_ = cv.putText(dispframe_, "7 = GARBAGE", (5, 160), cv.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1,
                            cv.LINE_AA)

    return dispframe_


def write_on_disk(queue, gtype_):
    frame_count = 0
    while queue.empty() is not True:
        a = queue.get()
        file_name = "recording" + os.path.splitext(os.path.basename(str(sys.argv[1])))[0] + "_" + gtype_ + "_frame"
        frame_count += 1
        # write a on disk
        while os.path.exists(DATA_PATH + '%s/%s%d.png' % (gtype_, file_name, frame_count)):
            frame_count += 1
        cv.imwrite(DATA_PATH + '%s/%s%d.png' % (gtype_, file_name, frame_count), a)


class KeySwitch:

    def keymanager(self, key):
        default_key = False, "garbage"
        return getattr(self, 'case_' + chr(key), lambda: default_key)()

    def case_0(self):
        return True, "oesophagus"

    def case_1(self):
        return True, "junction"

    def case_2(self):
        return True, "grande courbure gastrique"

    def case_3(self):
        return True, "pylore-antre"

    def case_4(self):
        return True, "angle"

    def case_5(self):
        return True, "retro vision"

    def case_6(self):
        return True, "sur angulaire-petite courbure"

    def case_7(self):
        return True, "garbage"


if __name__ == '__main__':
    frame_count = 0
    capture_counter = 10
    capture_flag = False
    gtype = "garbage"
    cap = cv.VideoCapture(INPUT_PATH)
    temp_frame_queue = Queue()
    while not cap.isOpened():
        cap = cv.VideoCapture(INPUT_PATH)
        cv.waitKey(100)
    while True:
        k = cv.waitKey(10) & 0xFF
        if k == ord('p'):
            while True:
                if cv.waitKey(1) & 0xFF == ord('s'):
                    break
        if k == ord('q'):
            break
        ks = KeySwitch()
        pressed, temp_type = ks.keymanager(k)
        if pressed:
            gtype = temp_type
            capture_flag = True
            write_on_disk(temp_frame_queue, gtype)

        ret, frame = cap.read()
        temp_frame_queue.put(frame)
        if temp_frame_queue.qsize() > 10:
            temp_frame_queue.get()
        if capture_flag:
            capture_counter -= 1
            if capture_counter == 0:
                capture_flag = False
                capture_counter = 10
                write_on_disk(temp_frame_queue, gtype)

        if ret is not True:
            break
        dispframe = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
        dispframe = textonframe(dispframe)
        cv.imshow('comparison', dispframe)
