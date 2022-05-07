import cv2 as cv
import os
import numpy as np
import tkinter
import tensorflow as tf
from queue import *
from tkinter import filedialog, messagebox


def text_on_frame(dispframe_, site, top2):
    """
    Adds informations on screen for user
    :param site
    :param dispframe_: frame on which the information have to be added
    :return: updated frame
    """
    dispframe_ = cv.putText(dispframe_, f"TOP1_pred = {site}", (5, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1,
                            cv.LINE_AA)
    dispframe_ = cv.putText(dispframe_, f"TOP2_pred = {top2}", (5, 80), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1,
                            cv.LINE_AA)
    return dispframe_


def crop(img):
    """
    Crop the frame for accurate DNN classification
    :param img: to be cropped
    :return: cropped image
    """
    size = img.shape
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


def video(cap_):
    """
    Read and display on screen the given video flux
    :param cap_: video capture input
    """
    score_queue = Queue()
    count = 0
    site = "null"
    top2 = "null"
    while True:
        ret, frame = cap_.read()
        k = cv.waitKey(1) & 0xFF
        if k == ord('p'):
            while True:
                k = cv.waitKey(1) & 0xFF
                if k == ord('s'):
                    break
        if k == ord('q') or not ret:
            break
        if count > 50 and not count % 5:
            if score_queue.qsize() > 5:
                score_queue.get()
            a = model.predict(tf.image.resize(cv.cvtColor(crop(frame), cv.COLOR_BGR2RGB), (224, 224))[None])
            score_queue.put(a)
            pred = np.sum(list(score_queue.queue),axis=0)
            site = labels[np.argmax(pred)]
            print(pred)
            pred[0][np.argmax(pred)] = 0
            top2 = labels[np.argmax(pred)]
        disp_frame = text_on_frame(cv.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC), site, top2)
        cv.imshow('classification', disp_frame)
        writer.write(disp_frame)
        count += 1
    cv.destroyWindow('classification')
    writer.release()

if __name__ == '__main__':

    MODEL_PATH = 'MODEL_MOBILENETV3'
    labels = ['angle', 'greater_curvature', 'junction', 'oesophagus', 'pylore_antre', 'retro_vision', 'unqualified']
    model = tf.keras.models.load_model(MODEL_PATH)
    print('model loaded')

    while True:
        tkinter.Tk().withdraw()
        tkinter.messagebox.showinfo('file path', 'choose the video you want to process')
        INPUT_PATH = filedialog.askopenfilename()
        base_name = os.path.splitext(os.path.basename(INPUT_PATH))[0]
        if os.path.splitext(INPUT_PATH)[1] != ".mov":
            print("Invalid file format")
        else:
            cap = cv.VideoCapture(INPUT_PATH)
            while not cap.isOpened():
                cap = cv.VideoCapture(INPUT_PATH)
                cv.waitKey(100)
            writer = cv.VideoWriter(f"classification_{MODEL_PATH}_{os.path.splitext(INPUT_PATH)[0][-3:]}.avi",
                                     cv.VideoWriter_fourcc(*"MJPG"), 24, (960, 540))

            video(cap)
            cap.release()
        ans = tkinter.messagebox.askyesno(title="options", message="You finished the processing of video %s \n"
                                                                   "Do you wish to continue ?" % base_name)
        if not ans:
            break
