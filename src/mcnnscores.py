import argparse
import os
import sys
import shutil
import time
import cv2
import gc

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import imagedata
import numpy as np
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__"))

class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch, num_classes):
        super(FineTuneModel, self).__init__()

        if arch.startswith('alexnet') :
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'alexnet'
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'vgg16'
        elif arch.startswith('vgg19'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'vgg19'
        elif arch.startswith('resnet18') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(512, num_classes)
            )
            self.modelName = 'resnet18'
        elif arch.startswith('resnet34') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(512, num_classes)
            )
            self.modelName = 'resnet34'
        elif arch.startswith('resnet50') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(2048, num_classes)
            )
            self.modelName = 'resnet50'
        elif arch.startswith('resnet101') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(2048, num_classes)
            )
            self.modelName = 'resnet101'
        elif arch.startswith('resnet152') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(2048, num_classes)
            )
            self.modelName = 'resnet152'
        elif arch.startswith('densenet121') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(512, num_classes)
            )
            self.modelName = 'densenet121'
        elif arch.startswith('densenet161') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(512, num_classes)
            )
            self.modelName = 'densenet161'
        elif arch.startswith('densenet169') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(81536, num_classes)
            )
            self.modelName = 'densenet169'
        elif arch.startswith('densenet201') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(94080, num_classes)
            )
            self.modelName = 'densenet201'
        else :
            raise("Finetuning not supported on this architecture yet")

        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False


    def forward(self, x):
        f = self.features(x)
        if self.modelName == 'alexnet' :
            f = f.view(f.size(0), 256 * 6 * 6)
            y = self.classifier[0](f)
            y = self.classifier[1](y)
            y = self.classifier[2](y)
            y = self.classifier[3](y)
            y = self.classifier[4](y)
            y = self.classifier[5](y)
        elif self.modelName == 'vgg16':
            f = f.view(f.size(0), -1)
            y = self.classifier[0](f)
            y = self.classifier[1](y)
            y = self.classifier[2](y)
            y = self.classifier[3](y)
            y = self.classifier[4](y)
            y = self.classifier[5](y)
        elif self.modelName == 'vgg19':
            f = f.view(f.size(0), -1)
            y = self.classifier[0](f)
            y = self.classifier[1](y)
            y = self.classifier[2](y)
            y = self.classifier[3](y)
            y = self.classifier[4](y)
            y = self.classifier[5](y)
        elif self.modelName == 'resnet18' :
            f = f.view(f.size(0), -1)
            y = f
        elif self.modelName == 'resnet34' :
            f = f.view(f.size(0), -1)
            y = f
        elif self.modelName == 'resnet50' :
            f = f.view(f.size(0), -1)
            y = f
        elif self.modelName == 'resnet101' :
            f = f.view(f.size(0), -1)
            y = f
        elif self.modelName == 'resnet152' :
            f = f.view(f.size(0), -1)
            y = f
        elif self.modelName == 'densenet121' :
            f = f.view(f.size(0), -1)
            y = f
        elif self.modelName == 'densenet161' :
            f = f.view(f.size(0), -1)
            y = f
        elif self.modelName == 'densenet169' :
            f = f.view(f.size(0), -1)
            y = f
        elif self.modelName == 'densenet201' :
            f = f.view(f.size(0), -1)
            y = f

        print('feature size: {}'.format(y.size()))
        return y

def extract_cnnfeat(videopath, model, args_batch_size, args_workers):

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        imagedata.ImageFolder(videopath, 1.0, transforms.Compose([
            #transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args_batch_size, shuffle=False,
        num_workers=args_workers, pin_memory=True)

    outfeats = None
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        
        output = model(input_var)
        outarray = output.data.cpu().numpy()
        if i == 0:
            outfeats = outarray
        else:
            outfeats = np.concatenate((outfeats, outarray), axis=0)

    model = None
    del model
    torch.cuda.empty_cache()


    return outfeats

def detect_duplication(feats):
    num = feats.shape[0]
    dim = feats.shape[1]

    dmethod = 0
    epsilon = 0.05

    distMat = np.zeros((num, num))
    
    mindist = 10e+6
    min_i = -1
    min_j = -1

    continue_scores = []
    for i in range(num-1):
        for j in range(i+1, num, 1):
            dist = 0
            if dmethod == 0:
                dist = np.linalg.norm((feats[i, :] - feats[j, :]), ord=2)
            elif dmethod == 1:
                dist = np.linalg.norm((feats[i, :] - feats[j, :]), ord=1)
            else:
                mu_u = [np.mean(feats[i,:])]*feats.shape[1]
                mu_v = [np.mean(feats[j,:])]*feats.shape[1]
                dotval = np.dot(feats[i,:] - mu_u, feats[j,:] - mu_v)
                u_norm = np.linalg.norm((feats[i,:] - mu_u), ord=2) 
                v_norm = np.linalg.norm((feats[j,:] - mu_v), ord=2) 
                dist = dotval/(u_norm * v_norm)
               
                if j == i+1:
                    dist = dist - 0.2
                else:
                    dist = dist - 0.2/(j-i)

            distMat[i,j] = dist
            if j-i == 1:
                continue_scores.append(dist)
            if j-i > 8 and dist < mindist:
                mindist = dist
                min_i = i
                min_j = j

    gmaxdist = np.max(distMat)
    gmindist = np.min(distMat)
    distMat = (distMat - gmindist)/(gmaxdist - gmindist)
    minDist = (mindist - gmindist)/(gmaxdist - gmindist)
    
    continue_scores = [dist/gmaxdist for dist in continue_scores]

    kl = kr = 0
    
    print('(min_i, min_j) = {}'.format([min_i, min_j]))
    #left shift
    kw = 0
    while min_i - kw >= 0:
        if abs(distMat[min_i-kw, min_j-kw] - minDist) < epsilon:
            kl = kw
#        else:
#            break

        kw += 1

    #right shift
    kw = 0
    while min_j + kw <  num:
        if abs(distMat[min_i+kw, min_j+kw] - minDist) < epsilon:
            kr = kw
#        else:
#            break

        kw += 1

    if min_i+kr > min_j-kl:
        kl = kr = 0

    print('kl = ' + str(kl) + ' , kr = ' + str(kr))

    num_copy_frames = kl + kr + 1
    confscore = -minDist/(num_copy_frames*(min_j - min_i))
    print('mindist = ' + str(minDist) + ', confscore = ' + str(confscore) + ', num_copy_frames: ' +
            str(num_copy_frames))

    mask_ranges = []
    mask_scores = []
    
    if num_copy_frames > 3:
        mask_ranges = [[min_i-kl, min_i+kr], [min_j-kl, min_j+kr]]
        if min_i-kl-1 < 0:
            mask_scores.append([0, distMat[min_j-kl, min_j-kl+1]])
        else:
            mask_scores.append([distMat[min_i-kl-1, min_i-kl], distMat[min_j-kl, min_j-kl+1]])
        
        if min_j+kr+1 >= num:
            mask_scores.append([distMat[min_i+kr-1, min_i+kr], 0])
        else:
            mask_scores.append([distMat[min_i+kr-1, min_i+kr], distMat[min_j+kr, min_j+kr+1]])

    print('mask_ranges: {}'.format(mask_ranges))
    print('mask_scores: {}'.format(mask_scores))
    return num, confscore, mask_ranges, mask_scores, continue_scores



def detect_copypaste(videopath, arch):
    # create model
    args_arch = arch #'resnet152'
    args_batch_size = 32 #512
    args_workers = 4
    num_classes = 2
    original_model = models.__dict__[args_arch](pretrained=True)
    model = FineTuneModel(original_model, args_arch, num_classes)


    #if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    if args_arch.startswith('alexnet') or args_arch.startswith('vgg'): # or args.arch.startswith('resnet'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # switch to evaluate mode
    model.eval()

    print('process video:  ' + str(videopath))
    
    cnnfeats = extract_cnnfeat(videopath, model, args_batch_size, args_workers)

    print('videofeats.shape: {}'.format(cnnfeats.shape))

    return detect_duplication(cnnfeats)


def extract_continue_scores(videopath, arch):
    # create model
    args_arch = arch #'resnet152'
    args_batch_size = 32
    args_workers = 4
    num_classes = 2
    original_model = models.__dict__[args_arch](pretrained=True)
    model = FineTuneModel(original_model, args_arch, num_classes)


    #if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    if args_arch.startswith('alexnet') or args_arch.startswith('vgg'): # or args.arch.startswith('resnet'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # switch to evaluate mode
    model.eval()

    print('process video:  ' + str(videopath))
    
    feats = extract_cnnfeat(videopath, model, args_batch_size, args_workers)
    num = feats.shape[0]
    dim = feats.shape[1]
    dmethod = 0

    continue_scores = []
    gmaxdist = 0
    for i in range(num-1):
        for j in range(i+1, num, 1):
            dist = 0
            if dmethod == 0:
                dist = np.linalg.norm((feats[i, :] - feats[j, :]), ord=2)
            elif dmethod == 1:
                dist = np.linalg.norm((feats[i, :] - feats[j, :]), ord=1)
            else:
                mu_u = [np.mean(feats[i,:])]*feats.shape[1]
                mu_v = [np.mean(feats[j,:])]*feats.shape[1]
                dotval = np.dot(feats[i,:] - mu_u, feats[j,:] - mu_v)
                u_norm = np.linalg.norm((feats[i,:] - mu_u), ord=2) 
                v_norm = np.linalg.norm((feats[j,:] - mu_v), ord=2) 
                dist = dotval/(u_norm * v_norm)
               
                if j == i+1:
                    dist = dist - 0.2
                else:
                    dist = dist - 0.2/(j-i)

            if dist > gmaxdist:
                gmaxdist = dist

            if j-i == 1:
                continue_scores.append(dist)

    continue_scores = [ dist/gmaxdist for dist in continue_scores]
    
    return continue_scores


def main(videopath, arch):
    #num, confscore, mask_ranges, mask_scores, continue_scores = detect_copypaste(videopath, arch)
    #print('num=' + str(num) + ', confscore={}'.format(confscore) + ', mask_range={}'.format(mask_ranges) + ', mask_scores={}'.format(mask_scores))
    continue_scores = detect_copypaste(videopath, arch)
    print('continue_scores: {}'.format(continue_scores))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
