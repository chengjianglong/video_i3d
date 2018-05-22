# Copyright 2018 Kitware Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Loads a sample video and classifies using a trained Kinetics checkpoint."""
import numpy as np
import sys
import os
import cv2
import tensorflow as tf

import i3dscores
import mcnnscores

def process_run_with_optout(videopath):
    vidname = videopath.split('/')[-1]
    video_fname = vidname.split('\n')[0]

    import copy

    algorithom = {
            "name": "kwi3d-framedrop",
            "version": "0.1.2",
            "description": "Frame drop detection based on I3D neural network. We propose a new approach for forensic analysis by exploiting the local spatio-temporal relationships within a portion of a video to robustly detect frame removals. A I3D Neural Network is adapted for frame drop detection. In order to further suppress the errors due by the network, we produce a refined video-level confidence score and demonstrate that it is superior to the raw output scores from the network",
            "metadata_usage": ["no-metadata"],
            "target_manipulations": ["SelectCutFrames"],
            "algorithm_type": ["integrity-indicator"],
            "indicator_type": ["digital"], 
            "media_type": ["video"], 
            "file_type": ["video/avi"], 
            "gpu_usage": ["required"], 
            "ram_usage": 1024,
            "expected_runtime": 100000,
            "code_link": "https://gitlab.mediforprogram.com/kitware/kinematic-authentication-of-video"
            }

    
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
                    "target_manipulations": ["SelectCutFrames"], 
                    "explanation": "consistency mismatchs mismatch occured based on the video-level confidence score", 
                    "video_localization": {
                        "frame_detection": frame_detection, 
                        "frame_optout": []
                    }
                }
            },
            "supplemental_information": None}

    return output

def process_run(scores, continue_scores, videopath):
    vidname = videopath.split('/')[-1]
    video_fname = vidname.split('\n')[0]
    scores = np.array(scores)

    import copy

    dist_scores = np.array(continue_scores[8:-8])

    usescores = 0.05*dist_scores + scores[:,1] + scores[:,2] - 0.1*scores[:,0]
    detection_score = np.max(usescores)

    if np.max(scores[:,1]) > 500:
        return process_run_with_optout(videopath)

    sdata = copy.deepcopy(usescores)
    sdata.sort()
    threshval = sdata[int(sdata.shape[0]*0.98)]

    framenum = scores.shape[0]
    peak_ids = []
    peak_scores = []
    for k in range(1, framenum-1, 1):
        if scores[k,1] >= scores[k,0] and scores[k,1] >= scores[k,2]:
            if scores[k,1]+scores[k,2] > scores[k-1,1]+scores[k-1,2] and scores[k,1]+scores[k,2] > scores[k+1,1]+scores[k+1,2]:
                peak_ids.append(k+9)
                peak_scores.append(usescores[k])

        if scores[k,2] >= scores[k,0] and scores[k,2] >= scores[k,1]:
            if scores[k,1]+scores[k,2] > scores[k-1,1]+scores[k-1,2] and scores[k,1]+scores[k,2] > scores[k+1,1]+scores[k+1,2]:
                peak_ids.append(k+9)
                peak_scores.append(usescores[k])


    if len(peak_ids) >= 10:
        return process_run_with_optout(videopath)


    threshval = 10000
    if len(peak_scores) > 0:
        sscores = sorted(peak_scores)
        threshval = sscores[int(len(sscores)*0.6)]
    
    drop_ids = []
    drop_scores = []
    wind = 8
    k = 0
    while k < len(peak_ids):
        kp = k+1
        best_k = k
        while kp < len(peak_ids) and peak_ids[kp] - peak_ids[k] < wind:
            idx1 = peak_ids[kp] - 9
            idx2 = peak_ids[k] - 9
            if usescores[idx1] > usescores[idx2]:
                best_k = kp
            
            kp += 1

        idxk = peak_ids[best_k] - 9
        if usescores[idxk] >= threshval:
            drop_ids.append(peak_ids[best_k])
            drop_scores.append(peak_scores[best_k])

        k = kp


    frame_detection = []
    if len(drop_ids) > 0:
        for i in range(len(drop_ids)):
            tmp = {
                    "frame_range":{ "start": drop_ids[i], "end": drop_ids[i]+1},
                    "scroe": drop_scores[i]}
            frame_detection.append(tmp)



    # write algorithom section.
    algorithm = {
            "name": "kwi3d-framedrop",
            "version": "0.1.2",
            "description": "Frame drop detection based on I3D neural network. We propose a new approach for forensic analysis by exploiting the local spatio-temporal relationships within a portion of a video to robustly detect frame removals. A I3D Neural Network is adapted for frame drop detection. In order to further suppress the errors due by the network, we produce a refined video-level confidence score and demonstrate that it is superior to the raw output scores from the network",
            "metadata_usage": ["no-metadata"],
            "target_manipulations": ["SelectCutFrames"],
            "algorithm_type": ["integrity-indicator"],
            "indicator_type": ["digital"], 
            "media_type": ["video"], 
            "file_type": ["video/avi"], 
            "gpu_usage": ["required"], 
            "ram_usage": 1024,
            "expected_runtime": 100000,
            "code_link": "https://gitlab.mediforprogram.com/kitware/kinematic-authentication-of-video"
            }
   
    output = {
            'algorithm': algorithm,
            "detection": {
                "input_filename": video_fname,
                "indicator_score": detection_score,
                "confidence": detection_score,
                "output": "OptOutLocalization", 
                "specificity": [ "global" ],
                "target_manipulations": ["frame-drop"], 
                "explanation": "consistency mismatchs mismatch occured based on the video-level confidence score",
                "video_localization": {
                    "frame_detection": frame_detection,
                    "frame_optout": [{"frame_range": { "start": 1, "end": 9 }}, {"frame_range": {"start": usescores.shape[0]+9, "end": usescores.shape[0]+17 }}] 
                }
            },
            "supplemental_information": {
                    "name": "peaks",
                    "description": "peaks determined by the output confidence scores among the whole video",
                    "value": {
                        "peak_threshold": threshval, 
                        "peak_ids": peak_ids,
                        "peak_scores": peak_scores,
                        }
            }
        }

    return output



def main(videopath):
    cap = cv2.VideoCapture(videopath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 100:
        return process_run_with_optout(videopath)
    
    frmcnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            frmcnt += 1
        else:
            break
   
    # Opt out videos with fps > 100 or frame count smaller than 17.
    if frmcnt < 17:
        return process_run_with_optout(videopath)
    
    
    continue_scores = mcnnscores.extract_continue_scores(videopath, 'resnet152')
    #print('continue_scores: {}'.format(continue_scores))
   

    tf.reset_default_graph()
    infilename = i3dscores.generate_i3d_infile(videopath)
    scores = i3dscores.eval_i3d_comb(infilename)
#
    return process_run(scores, continue_scores, videopath)
#


if __name__ == '__main__':
    print(main(sys.argv[1]))
