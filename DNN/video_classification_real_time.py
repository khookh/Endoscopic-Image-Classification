import cv2 as cv
import os
import numpy as np
import time
import tkinter
import tensorflow as tf
import queue as qq
import matplotlib.pyplot as plt
from tkinter import filedialog, messagebox
from multiprocessing import Process, Queue, Value


def row_normalization(row):
    i = np.argmax(row)
    temp_row = np.zeros(len(row))
    temp_row[i] = i + 1
    return temp_row


def save_time_line(predictions):
    x = np.arange(len(predictions)) / 25  # 25 frames per second
    predictions.apply(row_normalization)
    plt.figure(figsize=(12, 4))
    plt.plot(x, predictions)
    plt.xlabel('seconds')
    plt.ylabel('anatomical site')
    plt.title('time line - video classification')
    ax = plt.gca()
    ax.legends(['angle', 'corpus', 'junction', 'oesophagus', 'pylore_antre', 'retro_vision', 'unclassified'])
    plt.savefig('plot.png')


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


def classification_frames(queue_to, queue_treated, v):
    labels = ['angle', 'corpus', 'junction', 'oesophagus', 'pylore_antre', 'retro_vision', 'unclassified']
    predictions = np.array([0, 0, 0, 0, 0, 0, 0])
    site = ""
    local_count = 0
    score_queue = qq.Queue()
    model = tf.keras.models.load_model('DENSENET121')
    print('mmmm')
    while v.value != 1:
        while queue_to.empty():
            print(v.value)
            time.sleep(0)  # thread yield
            if v.value != 1:
                break
        if v.value != 1:
            break
        frame = queue_to.get()  # get frame from queue
        if local_count > 50 and not local_count % 5:
            if score_queue.qsize() > 5:
                score_queue.get()
            a = model.predict(tf.image.resize(cv.cvtColor(crop(frame), cv.COLOR_BGR2RGB), (224, 224))[None])
            score_queue.put(a)
            pred = np.sum(list(score_queue.queue), axis=0)
            predictions = np.append(predictions, pred, axis=0)
            site = labels[np.argmax(pred)]
        disp_frame = cv.putText(frame, f"TOP1_pred = {site}", (5, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                                1, cv.LINE_AA)
        queue_treated.put(disp_frame)
        local_count += 1
    save_time_line(predictions)


def video(INPUT_PATH, queue_to, v):
    """
    Read and display on screen the given video flux
    :param cap_: video capture input
    """
    cap = cv.VideoCapture(INPUT_PATH)
    while not cap.isOpened():
        cap = cv.VideoCapture(INPUT_PATH)
        cv.waitKey(100)
    ret, frame = cap.read()
    while ret and v.value != 1:
        while queue_to.qsize() > 20:
            time.sleep(0)
        ret, frame = cap.read()
        queue_to.put(frame)
    v.value = 1
    cap.release()


def display(queue_treated, v):
    while v.value != 1:
        while queue_treated.empty():
            time.sleep(0)  # thread yiel
        frame = queue_treated.get()  # get frame from queue
        cv.imshow('classification', cv.resize(frame, None, fx=0.7, fy=0.7, interpolation=cv.INTER_CUBIC))
        k = cv.waitKey(1) & 0xFF
        if k == ord('p'):
            while True:
                k = cv.waitKey(1) & 0xFF
                if k == ord('s'):
                    break
        if k == ord('q'):
            v.value = 1
            break
    cv.destroyWindow('classification')


if __name__ == '__main__':

    while True:
        tkinter.Tk().withdraw()
        tkinter.messagebox.showinfo('file path', 'choose the video you want to process')
        INPUT_PATH = filedialog.askopenfilename()
        base_name = os.path.splitext(os.path.basename(INPUT_PATH))[0]
        if os.path.splitext(INPUT_PATH)[1] != ".mov":
            print("Invalid file format")
        else:
            v = Value('i', 0)  # end flag for treatment process
            queue_to_process = Queue()
            queue_treated = Queue()
            image_classification = Process(target=classification_frames, args=(queue_to_process, queue_treated, v,))
            image_fetch = Process(target=video, args=(INPUT_PATH, queue_to_process, v,))
            image_display = Process(target=display, args=(queue_treated, v,))

            image_fetch.start()
            image_classification.start()
            image_display.start()

            image_fetch.join()

        ans = tkinter.messagebox.askyesno(title="options", message="You finished the processing of video %s \n"
                                                                   "Do you wish to continue ?" % base_name)
        if not ans:
            break
