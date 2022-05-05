import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 as cv
import os
import random
import sys

batch_size = 32
img_height = 500
img_width = 500

#INPUT_PATH = 'D:\\Thesis\\DataSet_cropped'
INPUT_PATH = 'D:\\Thesis\\kvasir-dataset-v2\\train'
OUTPUT_PATH = 'D:\\Thesis\\Heatmaps'

tf.config.list_physical_devices('GPU')
MODEL_PATH = 'MODEL_INCEPTIONV3'
labels = ['angle', 'greater_curvature', 'junction', 'oesophagus', 'pylore_antre', 'retro_vision', 'unqualified']

model = tf.keras.models.load_model(MODEL_PATH)
inceptionv3 = model.layers[3]

def text_on_frame(dispframe_, site):
    """
    Adds informations on screen for user
    :param site
    :param dispframe_: frame on which the information have to be added
    :return: updated frame
    """
    dispframe_ = cv.putText(dispframe_, f"ANAT SAT = {site}", (5, 20), cv.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1,
                            cv.LINE_AA)
    return dispframe_


def heatmap_from_pred(x):
    with tf.GradientTape() as tape:
        last_conv_layer = inceptionv3.get_layer('conv2d_93')#('conv5_block16_concat')#('conv5_block16_2_conv')
        iterate = tf.keras.models.Model([inceptionv3.inputs], [inceptionv3.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(x)
        #print(model_out.shape,model_out)
        class_out = model_out[:,:,: np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
    return tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)


def heatmap_img(img, sauce):
    heatmap = heatmap_from_pred(img[None])
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape((14, 14))
    heatmap = cv.resize(heatmap, (sauce.shape[1], sauce.shape[0]))
    heatmap = cv.applyColorMap(np.uint8(255 * heatmap), cv.COLORMAP_JET)
    heatmap = heatmap * 0.5 + sauce
    return heatmap


if __name__ == '__main__':
    quit = False
    while True:
        if quit:
            break
        picked_site = random.choice(os.listdir(INPUT_PATH))
        picked_img = random.choice(os.listdir(INPUT_PATH + "\\" + picked_site))
        img = cv.imread(INPUT_PATH + "\\" + picked_site + "\\" + picked_img)
        img_to = tf.image.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (img_height, img_width))
        pred = model.predict(img_to[None])
        site = labels[np.argmax(pred)]

        hm = heatmap_img(img_to, img)/255

        #img = cv.hconcat([img, hm])
        while True:
            k = cv.waitKey(1) & 0xFF
            cv.imshow('classification', text_on_frame(cv.resize(hm, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC), site))
            if k == ord('p'):
                break
            if k == ord('q'):
                quit = True
                break
            if k == ord('s'):
                frame_count = 0
                p = OUTPUT_PATH+"\\KVASIR_"+picked_site+"_"+site
                while os.path.exists(f'{p}_{frame_count}.png'):
                    frame_count += 1
                cv.imwrite(f'{p}_{frame_count}.png',hm*255)
                cv.imwrite(f'{p}_{frame_count}_blk.png', img)
                print(f'image saved as {p}')
                break

        cv.destroyWindow('classification')


