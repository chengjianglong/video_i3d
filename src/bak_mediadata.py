import numpy as np
import math
import sys
import concurrent.futures

import copy
import pandas as pd
import os
import os.path
from os.path import join

import matplotlib
matplotlib.use("Pdf")

import skimage.io as iio
import skvideo.io as vio
from skimage import img_as_ubyte
from skimage.transform import resize
from skimage.transform import rotate
import cv2

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def caloptflow(prev_frame, next_frame):
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def extractdata(vidname, start_fno, augop):
    directory = vidname.split('.avi')[0]
    directory = directory.split('.mp4')[0]
    directory = directory.replace('data', 'framedata')
    
    for fid in range(start_fno-1, start_fno+16, 1):
        if os.path.isfile(directory + '/' + str(fid) + '.jpg') == False:
            print(directory + '/' + str(fid) + '.jpg does not exist')
            return None
    
    def imgproc(fid, opid):
        if opid == 0:
            return cv2.resize(iio.imread(directory + '/' + str(fid) + '.jpg'), (256, 256))
        elif opid == 1:
            return cv2.resize(iio.imread(directory + '/' + str(fid) + '.jpg')[:, ::-1, :], (256, 256))
        elif opid == 2:
            return cv2.resize(iio.imread(directory + '/' + str(fid) + '.jpg')[::-1, :, :], (256, 256))
        elif opid == 3:
            return cv2.resize(img_as_ubyte(rotate(iio.imread(directory + '/' + str(fid) + '.jpg'), 90, resize=True)), (256, 256))
        elif opid == 4:
            return cv2.resize(img_as_ubyte(rotate(iio.imread(directory + '/' + str(fid) + '.jpg'), 180, resize=True)), (256, 256))
        elif opid == 5:
            return cv2.resize(img_as_ubyte(rotate(iio.imread(directory + '/' + str(fid) + '.jpg'), 270, resize=True)), (256, 256))
        elif opid == 6:
            return cv2.resize(img_as_ubyte(rotate(iio.imread(directory + '/' + str(fid) + '.jpg'), 45, resize=True)), (256, 256))
        elif opid == 7:
            return cv2.resize(img_as_ubyte(rotate(iio.imread(directory + '/' + str(fid) + '.jpg'), 135, resize=True)), (256, 256))
        elif opid == 8:
            return cv2.resize(img_as_ubyte(rotate(iio.imread(directory + '/' + str(fid) + '.jpg'), 225, resize=True)), (256, 256))
        elif opid == 9:
            return cv2.resize(img_as_ubyte(rotate(iio.imread(directory + '/' + str(fid) + '.jpg'), 315, resize=True)), (256, 256))
    
#    if augop == 0:
#        imgs = np.array([cv2.resize(iio.imread(directory + '/' + str(fid) + '.jpg'), (256,256)) for fid in range(start_fno-1, start_fno+16, 1)])
#    elif augop == 1:
#        imgs = np.array([cv2.resize(iio.imread(directory + '/' + str(fid) + '.jpg')[:, ::-1, :], (256,256)) for fid in range(start_fno-1, start_fno+16, 1)])
#    elif augop == 2:
#        imgs = np.array([cv2.resize(iio.imread(directory + '/' + str(fid) + '.jpg')[::-1, :, :], (256,256)) for fid in range(start_fno-1, start_fno+16, 1)])
#    elif augop == 3:
#        imgs = np.array([cv2.resize(img_as_ubyte(rotate(iio.imread(directory + '/' + str(fid) + '.jpg'), 90, resize=True)), (256,256)) for fid in range(start_fno-1, start_fno+16, 1)])
#    elif augop == 4:
#        imgs = np.array([cv2.resize(img_as_ubyte(rotate(iio.imread(directory + '/' + str(fid) + '.jpg'), 180, resize=True)), (256,256)) for fid in range(start_fno-1, start_fno+16, 1)])
#    elif augop == 5:
#        imgs = np.array([cv2.resize(img_as_ubyte(rotate(iio.imread(directory + '/' + str(fid) + '.jpg'), 270, resize=True)), (256,256)) for fid in range(start_fno-1, start_fno+16, 1)])
#    elif augop == 6:
#        imgs = np.array([cv2.resize(img_as_ubyte(rotate(iio.imread(directory + '/' + str(fid) + '.jpg'), 45, resize=True)), (256,256)) for fid in range(start_fno-1, start_fno+16, 1)])
#    elif augop == 7:
#        imgs = np.array([cv2.resize(img_as_ubyte(rotate(iio.imread(directory + '/' + str(fid) + '.jpg'), 135, resize=True)), (256,256)) for fid in range(start_fno-1, start_fno+16, 1)])
#    elif augop == 8:
#        imgs = np.array([cv2.resize(img_as_ubyte(rotate(iio.imread(directory + '/' + str(fid) + '.jpg'), 225, resize=True)), (256,256)) for fid in range(start_fno-1, start_fno+16, 1)])
#    elif augop == 9:
#        imgs = np.array([cv2.resize(img_as_ubyte(rotate(iio.imread(directory + '/' + str(fid) + '.jpg'), 315, resize=True)), (256,256)) for fid in range(start_fno-1, start_fno+16, 1)])
#   
#
#    flows = []
#    for i in range(imgs.shape[0]-1):
#        tmp_flow = caloptflow(imgs[i], imgs[i+1])
#        flows.append(tmp_flow)
#    flows = np.array(flows)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        imgs = np.array([tmp_img for tmp_img in executor.map(imgproc, range(start_fno-1,
            start_fno+16, 1), [augop]*17)])
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        flows = np.array([tmp_flow for tmp_flow in executor.map(caloptflow, imgs[0:-2],
            imgs[1:-1])])
    
    imgs = imgs[1:,16:240, 16:240, :]
    #print('itemdata with rgb and optflow:')
    #print(imgs.shape)

    flows = flows[:,16:240, 16:240, :]
    #print(flows.shape)
    
    return [imgs, flows]

def extract_check_data(vidname, start_fno, augop, b_vid2img, tmpfolder):
    directory = tmpfolder
    if b_vid2img == True:
        os.system('rm -rf ' + directory)
        os.system('mkdir ' + directory)
        cap = cv2.VideoCapture(vidname)
        imgno = -1
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                imgno += 1
                img = cv2.resize(frame, (256,256))
                imgpath = directory + '/' + str(imgno) + '.jpg'
                cv2.imwrite(imgpath, img)
            else:
                break


    for fid in range(start_fno-1, start_fno+16, 1):
        if os.path.isfile(directory + '/' + str(fid) + '.jpg') == False:
            print(directory + '/' + str(fid) + '.jpg does not exist')
            return None

    #print('check directory: {}'.format([directory + '/' + str(fid) + '.jpg' for fid in range(start_fno-1, start_fno+16,1)]))
    if augop == 0:
        imgs = np.array([iio.imread(directory + '/' + str(fid) + '.jpg') for fid in range(start_fno-1, start_fno+16, 1)])
    if augop == 1:
        imgs = np.array([iio.imread(directory + '/' + str(fid) + '.jpg')[:, ::-1, :] for fid in range(start_fno-1, start_fno+16, 1)])
    if augop == 2:
        imgs = np.array([iio.imread(directory + '/' + str(fid) + '.jpg')[::-1, :, :] for fid in range(start_fno-1, start_fno+16, 1)])
    if augop == 3:
        imgs = np.array([img_as_ubyte(rotate(iio.imread(directory + '/' + str(fid) + '.jpg'), 90, resize=True)) for fid in range(start_fno-1, start_fno+16, 1)])
    if augop == 4:
        imgs = np.array([img_as_ubyte(rotate(iio.imread(directory + '/' + str(fid) + '.jpg'), 180, resize=True)) for fid in range(start_fno-1, start_fno+16, 1)])
    if augop == 5:
        imgs = np.array([img_as_ubyte(rotate(iio.imread(directory + '/' + str(fid) + '.jpg'), 270, resize=True)) for fid in range(start_fno-1, start_fno+16, 1)])
    if augop == 6:
        imgs = np.array([img_as_ubyte(rotate(iio.imread(directory + '/' + str(fid) + '.jpg'), 45, resize=True)) for fid in range(start_fno-1, start_fno+16, 1)])
    if augop == 7:
        imgs = np.array([img_as_ubyte(rotate(iio.imread(directory + '/' + str(fid) + '.jpg'), 135, resize=True)) for fid in range(start_fno-1, start_fno+16, 1)])
    if augop == 8:
        imgs = np.array([img_as_ubyte(rotate(iio.imread(directory + '/' + str(fid) + '.jpg'), 225, resize=True)) for fid in range(start_fno-1, start_fno+16, 1)])
    if augop == 9:
        imgs = np.array([img_as_ubyte(rotate(iio.imread(directory + '/' + str(fid) + '.jpg'), 315, resize=True)) for fid in range(start_fno-1, start_fno+16, 1)])
    
    #print('augop: ' + str(augop) + ', imgs.shape: {}'.format(imgs.shape))
    flows = []
    for i in range(imgs.shape[0]-1):
        #print('caloptflow ' + str(i) + '-' + str(i+1))
        tmp_flow = caloptflow(imgs[i], imgs[i+1])
        flows.append(tmp_flow)
    flows = np.array(flows)

    imgs = imgs[1:,16:240, 16:240, :]
    #imgs = tf.convert_to_tensor(imgs[1:,16:240, 16:240, :])
    #print('itemdata with rgb and optflow:')
    #print(imgs.shape)

    flows = flows[:,16:240, 16:240, :]
    #flows = tf.convert_to_tensor(flows[:,16:240, 16:240, :])

    #print(flows.shape)
    
    return [imgs, flows]

def checkdata(vidname, start_fno):
    directory = vidname.split('.avi')[0]
    directory = directory.replace('data', 'framedata')

    for fid in range(start_fno-1, start_fno+16, 1):
        if os.path.isfile(directory + '/' + str(fid) + '.jpg') == False:
            print(directory + '/' + str(fid) + '.jpg does not exist')
            return False

    return True
                    

def LoadInputData(items, parttag):
    rgbs = []
    flows = []
    truths = []
    ino = 0

    tmpfolder = 'tmp_' + parttag
    if os.path.isdir(tmpfolder) == False:
        os.system('mkdir ' + tmpfolder)

    pre_vidname = ''
    b_vid2img = False
    for item in items:
        ino += 1
        elems = item.split(' ')
        #print(str(ino) + ', video path: ' + elems[0] + ', start_fno = ' + elems[1] + ', truth = ' + elems[2])
        tmp_label = [0, 0, 0]
        tmp_label[int(elems[2])] = 1
        augop = 0
        if len(elems) == 4:
            augop = int(elems[3])

        if elems[0] != pre_vidname:
            b_vid2img = True
            pre_vidname = elems[0]
        else:
            b_vid2img = False

        res = extract_check_data(elems[0], int(elems[1]), augop, b_vid2img, tmpfolder)
        if res is None:
            continue
        else:
            truths.append(tmp_label)
            rgbs.append(res[0])
            flows.append(res[1])

    rgbs = np.array(rgbs)
    flows = np.array(flows)
    truths = np.array(truths)

#    rgbs = tf.convert_to_tensor(rgbs)
#    flows = tf.convert_to_tensor(flows)
#    truths = tf.convert_to_tensor(truths)
#    print('Overall loaded data: ')
#    print(rgbs.shape)
#    print(flows.shape)

    return rgbs, flows, truths

def CheckInputData(items):
    valid_items = []
    ino = 0
    for item in items:
        ino += 1
        elems = item.split(' ')
        #print(str(ino) + ', video path: ' + elems[0] + ', start_fno = ' + elems[1] + ', truth = ' + elems[2])
        tmp_label = [0, 0, 0]
        tmp_label[int(elems[2])] = 1
        res = checkdata(elems[0], int(elems[1]))
        if res == True:
            valid_items.append(item)

    return valid_items

def LoadItems(infname):
    items = open(infname, 'r')
    out_items = []
    for item in items:
        item = item.split('\n')[0]
        out_items.append(item)

    return out_items


if __name__ == '__main__':
  items = LoadItems(sys.argv[1])
  LoadInputData(items, '')
