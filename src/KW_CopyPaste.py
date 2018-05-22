import argparse
import os
import sys
import cv2
import numpy as np

import i3dscores
import mcnnscores
import torch
import tensorflow as tf

def process_run_with_optout(videopath):
    vidname = videopath.split('/')[-1]
    video_fname = vidname.split('\n')[0]

    import copy

    algorithom = {
            "name": "kw-cnn-copypaste",
            "version": "0.1.2",
            "description": "Frame duplication detection based on the state-of-the-art Convolutional Neural Network (CNN) and I3D neural network. CNN is used to extract frame-level feature, and I3D neural network to distinguish between selected frames and duplicated frames.",
            "metadata_usage": ["no-metadata"],
            "target_manipulations": ["CopyPaste"], 
            "algorithm_type": ["integrity-indicator"],
            "indicator_type": ["digital"], 
            "media_type": ["video"], 
            "file_type": ["video/avi"], 
            "gpu_usage": ["required"], 
            "ram_usage": 1024,
            "expected_runtime": 100000,
            "code_link": "https://gitlab.mediforprogram.com/kitware/kinematic-authentication-of-video"
            }

    
    dupmask = []
    confscore = -10.0
    
    frame_detection = []
    


    # write algorithom section.
    output = {
            "algorithom": algorithom,
            "detection": {
                "detection": {
                    "input_filename": video_fname,
                    "indicator_score": confscore,
                    "confidence": confscore,
                    "output": "OptOutAll", 
                    "specificity": [ "global" ],
                    "target_manipulations": ["CopyPaste"], 
                    "explanation": "consistency mismatchs mismatch occured based on the video-level confidence score", 
                    "video_localization": {
                        "frame_detection": frame_detection, 
                        "frame_optout": []
                    }
                }
            },
            "supplemental_information": None}

    return output

def process_run(videopath, num, confscore, mask_ranges, mask_scores, continue_scores, i3d_scores):
    vidname = videopath.split('/')[-1]
    video_fname = vidname.split('\n')[0]

    import copy

    algorithom = {
            "name": "kw-cnn-copypaste",
            "version": "0.1.2",
            "description": "Frame duplication detection based on the state-of-the-art Convolutional Neural Network (CNN) and I3D neural network. CNN is used to extract frame-level feature, and I3D neural network to distinguish between selected frames and duplicated frames.",
            "metadata_usage": ["no-metadata"],
            "target_manipulations": ["CopyPaste"], 
            "algorithm_type": ["integrity-indicator"],
            "indicator_type": ["digital"], 
            "media_type": ["video"], 
            "file_type": ["video/avi"], 
            "gpu_usage": ["required"], 
            "ram_usage": 1024,
            "expected_runtime": 100000,
            "code_link": "https://gitlab.mediforprogram.com/kitware/kinematic-authentication-of-video"
            }

    scores = np.array(continue_scores[8:-8])
    if i3d_scores is not None:
        scores = 0.05*np.array(continue_scores[8:-8]) + np.array(i3d_scores)[:, 1] + np.array(i3d_scores)[:,2] - 0.1*np.array(i3d_scores)[:,0]
    
    dupmask = []
    dupscores = []
    if len(mask_ranges) > 0:
        m1 = mask_ranges[0][0]
        n1 = mask_ranges[0][1]
        m2 = mask_ranges[1][0]
        n2 = mask_ranges[1][1]

        if m1-9 >= 0 and n1-8 < len(scores) and m2-9 >= 0 and n2-8 < len(scores):
            mink1 = min(m1-9, m2-9)
            mink2 = len(scores) - 1 - max(n1-8, n2-8)
            wind = min(mink1, mink2)
            hwid = 2
            if wind > hwid:
                wind = hwid

            score1 = 0
            score2 = 0
            for k in range(-wind, wind+1, 1):
                score1 = scores[m1-9+k] + scores[n1-8+k]
                score2 = scores[m2-9+k] + scores[n2-8+k]

            print('score1 = ' + str(score1) + ', score2 = ' + str(score2))
            dupscores = [score1]
            dupmask = [[mask_ranges[0][0]+1, mask_ranges[0][1]+1]]
            if score1 < score2:
                dupmask =[ [mask_ranges[1][0]+1, mask_ranges[1][1]+1] ]
                dupscores = [score2]

            if m2 - n1 <= 0:
                if score1 > score2:
                    dupmask = [ [mask_ranges[0][0]+1, mask_ranges[0][1]+1], [mask_ranges[1][0]+1, mask_ranges[1][1]+1] ]
                    dupscores = [score1, score2]
                else:
                    dupmask = [ [mask_ranges[1][0]+1, mask_ranges[1][1]+1], [mask_ranges[0][0]+1, mask_ranges[0][1]+1] ]
                    dupscores = [score2, score1]
                    

    frame_detection = []
    for i in range(len(dupmask)):
        frame_detection.append({
            "frame_range": {"start": dupmask[i][0], "end": dupmask[i][1]},
            "score": float(dupscores[i])
        })

    # write algorithom section.
    output = {
            "algorithom": algorithom,
            "detection": {
                "detection": {
                    "input_filename": video_fname,
                    "indicator_score": confscore,
                    "confidence": confscore,
                    "output": "Processed", 
                    "specificity": [ "global" ],
                    "target_manipulations": ["CopyPaste"], 
                    "explanation": "consistency mismatchs mismatch occured based on the video-level confidence score", 
                    "video_localization": {
                        "frame_detection": frame_detection, 
                    "frame_optout": [
                        {"frame_range":{ "start": 1, "end": 9}},
                        {"frame_range":{ "start": num-8, "end": num }}]
                    }
                }
            },
            "supplemental_information": None}

    return output


def main(videopath):
    arch = 'resnet152'
    cap = cv2.VideoCapture(videopath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frmcnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            frmcnt += 1
        else:
            break
   
    # Opt out videos with fps > 100 or frame count smaller than 17.
    if fps > 100 or frmcnt < 17:
        return process_run_with_optout(videopath)
        

    num, confscore, mask_ranges, mask_scores, continue_scores = mcnnscores.detect_copypaste(videopath, arch)
    #print('num=' + str(num) + ', confscore={}'.format(confscore) + ', mask_range={}'.format(mask_ranges) + ', mask_scores={}'.format(mask_scores))
    #print('continue_scores: {}'.format(continue_scores))
    
    i3d_scores = None
    if len(mask_ranges) > 0:
        tf.reset_default_graph()
        infilename = i3dscores.generate_i3d_infile(videopath)
        i3d_scores = i3dscores.eval_i3d_comb(infilename)

    #print('i3d_scores: {}'.format(i3d_scores))
    return process_run(videopath, num, confscore, mask_ranges, mask_scores, continue_scores, i3d_scores)


if __name__ == '__main__':
    print(main(sys.argv[1]))
