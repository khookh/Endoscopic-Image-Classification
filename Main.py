import numpy as np
import cv2 as cv
import sys
import os

from multiprocessing import Queue

INPUT_PATH = sys.argv[1]
DATA_PATH = sys.argv[2]


def textonframe(dispframe_):
    """
    Adds informations on screen for user
    :param dispframe_: frame on which the information have to be added
    :return: updated frame
    """
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


def write_on_disk(frame_, gtype_):
    """
    :param frame_: frame to be written on disk
    :param gtype_: corresponding anatomical site
    """
    frame_count = 0
    file_name = "recording" + os.path.splitext(os.path.basename(str(sys.argv[1])))[0] + "_" + gtype_ + "_frame"
    frame_count += 1
    # write a on disk
    while os.path.exists(DATA_PATH + '%s/%s%d.png' % (gtype_, file_name, frame_count)):
        frame_count += 1
    cv.imwrite(DATA_PATH + '%s/%s%d.png' % (gtype_, file_name, frame_count), frame_)


class KeySwitch:
    """
    Implements a switch to match user key input with correct anatomical site
    """

    def keymanager(self, key):
        """
        :param key: key (ord value) user has pressed
        :return: Tuple ('has the user pressed?', 'which anatomical site?')
        """
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


def video(cap_):
    """
    Read and display on screen the given video flux
    :param cap_: video capture input
    """
    # counter used to get the 10th frame after user capture input
    capture_counter = 10
    # flag used to know if the user activated a capture event
    capture_flag = False
    # current / last observed observed site
    gtype = "garbage"
    # queue storing the last 10 frames
    temp_frame_queue = Queue()
    while True:
        ret, frame = cap_.read()
        temp_frame_queue.put(frame)

        k = cv.waitKey(1) & 0xFF
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
            # write the last 10th frame
            write_on_disk(temp_frame_queue.get(), gtype)
            # write the current frame
            write_on_disk(frame, gtype)

        if temp_frame_queue.qsize() > 10:
            # keep in track only the 10 last frames
            temp_frame_queue.get()

        if capture_flag:
            capture_counter -= 1
            if capture_counter == 0:
                capture_flag = False
                capture_counter = 10
                # write the 10th frame after capture
                write_on_disk(frame, gtype)

        if ret is not True:
            break
        dispframe = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
        dispframe = textonframe(dispframe)
        cv.imshow('comparison', dispframe)
    cap_.release()


if __name__ == '__main__':
    cap = cv.VideoCapture(INPUT_PATH)
    while not cap.isOpened():
        cap = cv.VideoCapture(INPUT_PATH)
        cv.waitKey(100)
    video(cap)
