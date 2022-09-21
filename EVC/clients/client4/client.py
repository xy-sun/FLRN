#-------------------------------------
# Project: Lightweight Industrial Image Classifier based on Federated Few-Shot Learning
# code is based on https://github.com/floodsung/LearningToCompare_FSL
#-------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator as tg
import os
import math
import argparse
import scipy as sp
import scipy.stats
from ecci_sdk import Client
import threading

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 32)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 2)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 1)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 1)
parser.add_argument("-e","--episode",type = int, default= 500000)
parser.add_argument("-t","--test_episode", type = int, default = 100)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
args = parser.parse_args()

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

def mini_imagenet_folders():
    metatrain_folder = './client4/train4'
    metaval_folder = './client4/train4'
    metatrain_folders = [os.path.join(metatrain_folder, label) \
                         for label in os.listdir(metatrain_folder) \
                         if os.path.isdir(os.path.join(metatrain_folder, label)) \
                         ]
    metaval_folders = [os.path.join(metaval_folder, label) \
                       for label in os.listdir(metaval_folder) \
                       if os.path.isdir(os.path.join(metaval_folder, label)) \
                       ]
    return metatrain_folders, metaval_folders

class CNNEncoder(nn.Module):
    """docstring for ClassName"""

    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out

def main():
    ecci_client = Client()
    mqtt_thread = threading.Thread(target=ecci_client.initialize)
    mqtt_thread.start()
    ecci_client.wait_for_ready()
    FEATURE_DIM = args.feature_dim
    CLASS_NUM = args.class_num
    SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
    BATCH_NUM_PER_CLASS = args.batch_num_per_class
    EPISODE = args.episode
    TEST_EPISODE = args.test_episode
    LEARNING_RATE = args.learning_rate
    GPU = args.gpu
    HIDDEN_UNIT = args.hidden_unit
    metatrain_folders,metatest_folders = mini_imagenet_folders()
    for episode in range(EPISODE):
        data_msg_queue = ecci_client.get_sub_data_payload_queue()
        data_msg = data_msg_queue.get()
        feature_encoder = data_msg['feature_encoder']
        task =tg.MiniImagenetTask(metatrain_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)  # task = tg.MiniImagenetTask(metatrain_folders,CLASS_NUM,5,10)
        sample_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train",
                                                             shuffle=False)
        batch_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=BATCH_NUM_PER_CLASS, split="test",
                                                            shuffle=True)
        samples,sample_labels = sample_dataloader.__iter__().next()
        batches,batch_labels = batch_dataloader.__iter__().next()
        sample_features = feature_encoder(Variable(samples).cuda(GPU))
        batch_features = feature_encoder(Variable(batches).cuda(GPU))
        payload = {"batch_labels": batch_labels, "sample_features": sample_features, "batch_features": batch_features}
        ecci_client.send_message(payload, "cloud")

if __name__ == '__main__':
    main()
