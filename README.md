# Synopsis
• We propose a 3D convolutional network for frame dropping detection, and the confidence score is defined with a peak detection trick and a scale term based on
the output score curves. It is able to identify whether frame dropping exists and even determine the exact location of frame dropping without any information of
the reference/original video. 

• For performance comparison, we also compare to a series of baselines including cue-based algorithms (Color histogram, motion energy, and optical flow)
and learning-based algorithms (an SVM algorithm and convolutional neural networks (CNNs) using two or three frames as input)

We extend the previous C3D, a single stream network, to two-stream 3D neural networks (I3D) to incorporate flow information as input. I3D has three advantages:

• Inflating 2D ConvNets into 3D. Filters are typically square and we just make them cubic – N × N filters
become N × N × N.

• Bootstrapping 3D filters from 2D Filters to bootstrap parameters from the pre-trained ImageNet models.

• Pacing receptive field growth in space, time and network depth.

For more information, please read the reference: https://arxiv.org/pdf/1705.07750.pdf

# Contributors
Chengjiang Long (chengjiang.long@kitware.com) -- algorithm development and implementation


## Directory Structure

<b>docs/</b> - documentation

<b>lib/</b> - 3rd party libraries needed

<b>src/</b> - Source code 

<b>test</b> - scripts to run the indicator on specified data and verify expected results. 

# Documentation Template:

## Synopsis

Integrate the code to produce a detection score for an input video. 

## Inputs / Outputs

Input: Images, Videos
Output: detection score

## Documentation Links


## Prerequisites

1. Install video-caffe
URL: https://github.com/chuckcho/video-caffe

2. Install Python+OpenCV

For the I3D python version, the requirements are: 

PyTorch

TensorFlow

tensorboard_logger

dm-sonnet

sk-video

scikit-image

opencv-python



## Installation

Go to the folder src
mkdir build
cd build
cmake ..
ccmake .  (Please point the Caffe_DIR to the "video-caffe/build"'s path)
make

For the I3D python version, to install the requirements, you can run:
pip install -r lib/i3d_requirements.txt

## Usage

(Assuming you are in the ``Integrity_Integrator`` folder.)

``$ cd test``

``$ ./run4nist.sh``

``$ ./test_kw_kinematics.sh``

and then you will see the confidence score for a video outputs on the screen.

## Tests

%%% The exact steps (e.g. input media and parameters) to produce an evaluation result or otherwise prove that the algorithm is working as expected. 
%%% Include code to verify that the output produced by the code is exactly what is expected.

(Assuming you are in the ``Integrity_Integrator`` folder.)

./src/build/nfdd-c3d-main4nist 21728369baaaae41f9d8aaadbd9edd5c.mp4

The expected output is given in Integrity_Indicator/expected_output.txt

For the I3D python version,

``$ cd test``

``$ python ../src/KW_SelectCutFrames.py  Forgery_white_Car.mp4``

``$ python ../src/KW_CopyPaste.py  Forgery_white_Car.mp4``


