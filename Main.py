import numpy as np
import cv2 as cv
import sys
import os

from multiprocessing import Queue

frame_count = 0


def textonframe(dispframe):
    dispframe = cv.putText(dispframe, "0 = oesophagus", (5, 20), cv.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1,
                           cv.LINE_AA)
    dispframe = cv.putText(dispframe, "1 = junction", (5, 40), cv.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1,
                           cv.LINE_AA)
    dispframe = cv.putText(dispframe, "2 = grande courbure gastrique", (5, 60), cv.FONT_HERSHEY_SIMPLEX, .4,
                           (0, 0, 255), 1,
                           cv.LINE_AA)
    dispframe = cv.putText(dispframe, "3 = pylore-antre", (5, 80), cv.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1,
                           cv.LINE_AA)
    dispframe = cv.putText(dispframe, "4 = angle", (5, 100), cv.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1,
                           cv.LINE_AA)
    dispframe = cv.putText(dispframe, "5 = retro vision", (5, 120), cv.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1,
                           cv.LINE_AA)
    dispframe = cv.putText(dispframe, "6 = sur angulaire-petite courbure", (5, 140), cv.FONT_HERSHEY_SIMPLEX, .4,
                           (0, 0, 255), 1,
                           cv.LINE_AA)
    dispframe = cv.putText(dispframe, "7 = GARBAGE", (5, 160), cv.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1,
                           cv.LINE_AA)

    return dispframe


def write_on_disk(queue, gtype_):
    global frame_count
    while queue.empty() is not True:
        a = queue.get()
        frame_count += 1
        # write a on disk
        while os.path.exists('E:/Gastroscopies/capture/%s/%s%d.png' % (gtype_, os.path.basename(str(sys.argv[1])), frame_count)):
            frame_count += 1
        cv.imwrite('E:/Gastroscopies/capture/%s/%s%d.png' % (gtype_, os.path.basename(str(sys.argv[1])), frame_count),a)


if __name__ == '__main__':
    frame_count = 0
    capture_counter = 10
    capture_flag = False
    gtype = "garbage"
    cap = cv.VideoCapture(str(sys.argv[1]))
    temp_frame_queue = Queue()
    while not cap.isOpened():
        cap = cv.VideoCapture(str(sys.argv[1]))
        cv.waitKey(100)
    while True:
        k = cv.waitKey(10) & 0xFF
        if k == ord('p'):
            while True:
                if cv.waitKey(1) & 0xFF == ord('s'):
                    break
        if k == ord('q'):
            break
        if k == ord('0'):
            capture_flag = True
            gtype = "oesophagus"
            write_on_disk(temp_frame_queue, gtype)
        if k == ord('1'):
            capture_flag = True
            gtype = "junction"
            write_on_disk(temp_frame_queue, gtype)
        if k == ord('2'):
            capture_flag = True
            gtype = "grande courbure gastrique"
            write_on_disk(temp_frame_queue, gtype)
        if k == ord('3'):
            capture_flag = True
            gtype = "pylore-antre"
            write_on_disk(temp_frame_queue, gtype)
        if k == ord('4'):
            capture_flag = True
            gtype = "angle"
            write_on_disk(temp_frame_queue, gtype)
        if k == ord('5'):
            capture_flag = True
            gtype = "retro vision"
            write_on_disk(temp_frame_queue, gtype)
        if k == ord('6'):
            capture_flag = True
            gtype = "sur angulaire-petite courbure"
            write_on_disk(temp_frame_queue, gtype)
        if k == ord('7'):
            capture_flag = True
            gtype = "garbage"
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
