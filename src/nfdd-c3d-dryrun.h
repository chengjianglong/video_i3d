#ifndef NFDD_C3D_DRYRUN_HEADER_H
#define NFDD_C3D_DRYRUN_HEADER_H

#include "opencv2/opencv.hpp"
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

struct video_detect_values
{
    float peak_thresh;
    vector<int> peak_ids;
    vector<float> peak_scores;

    float detection_score;
    vector<float> frame_scores;
    float frame_score_thresh;
    vector<int> frame_optout;
};

video_detect_values detect_frame_drop(string video_fname);

#endif

