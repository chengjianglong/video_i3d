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

def imgproc(fid, opid, directory):
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

def extractdata(vidname, start_fno, augop):
    directory = vidname.split('.avi')[0]
    directory = directory.split('.mp4')[0]
    directory = directory.replace('data', 'framedata')
    
    for fid in range(start_fno-1, start_fno+16, 1):
        if os.path.isfile(directory + '/' + str(fid) + '.jpg') == False:
            print(directory + '/' + str(fid) + '.jpg does not exist')
            return None
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        imgs = np.array([tmp_img for tmp_img in executor.map(imgproc, range(start_fno-1, start_fno+16, 1), [augop]*17, [directory]*17)])
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        flows = np.array([tmp_flow for tmp_flow in executor.map(caloptflow, imgs[0:-2],
            imgs[1:-1])])
    
    imgs = imgs[1:,16:240, 16:240, :]
    #print('itemdata with rgb and optflow:')
    #print(imgs.shape)

    flows = flows[:,16:240, 16:240, :]
    #print(flows.shape)
    
    return [imgs, flows]

def extract_check_data(vidname, start_fno, augop, b_vid2img, tmpfolder, pre_rgb, pre_flow):
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

    

    if b_vid2img == True:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            imgs = np.array([tmp_img for tmp_img in executor.map(imgproc, range(start_fno-1, start_fno+16, 1), [augop]*17, [directory]*17)])
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            flows = np.array([tmp_flow for tmp_flow in executor.map(caloptflow, imgs[0:-1], imgs[1:])])

        imgs = imgs[1:,16:240, 16:240, :]

        flows = flows[:,16:240, 16:240, :]

    else:
        imgs = pre_rgb[1:, :, :, :]
        flows = pre_flow[1:, :, :, :]
        img1 = imgproc(start_fno+14, augop, directory)
        img2 = imgproc(start_fno+15, augop, directory)
        
        tmp_img = np.array([img2[16:240, 16:240, :]])
        tmp_flow = np.array([caloptflow(img1, img2)[16:240, 16:240, :]])

        imgs = np.concatenate((imgs, tmp_img))
        flows = np.concatenate((flows, tmp_flow))

    return [imgs, flows]

def checkdata(vidname, start_fno):
    directory = vidname.split('.avi')[0]
    directory = directory.replace('data', 'framedata')

    for fid in range(start_fno-1, start_fno+16, 1):
        if os.path.isfile(directory + '/' + str(fid) + '.jpg') == False:
            print(directory + '/' + str(fid) + '.jpg does not exist')
            return False

    return True
                    

def LoadInputData(items, parttag, pre_vidname, pre_rgb, pre_flow):
    rgbs = []
    flows = []
    truths = []
    ino = 0

    tmpfolder = 'tmp_' + parttag
    if os.path.isdir(tmpfolder) == False:
        os.system('mkdir ' + tmpfolder)

    cur_vidname = ''
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
            pre_rgb = None
            pre_flow = None
        else:
            b_vid2img = False

        cur_vidname = elems[0]
        res = extract_check_data(elems[0], int(elems[1]), augop, b_vid2img, tmpfolder, pre_rgb, pre_flow)
        if res is None:
            continue
        else:
            pre_rgb = res[0]
            pre_flow = res[1]
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

    return rgbs, flows, truths, cur_vidname

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
