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
import task_generator_chip as tg
import os
import math
import argparse

from ecci_sdk import Client
import threading

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 32)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 2)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 1)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 1)
parser.add_argument("-e","--episode",type = int, default= 1000000)
parser.add_argument("-t","--test_episode", type = int, default = 100)
# parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
args = parser.parse_args()

def omniglot_character_folders():
    metatrain_folder = './new_client/test'
    metatest_folder = './new_client/test'
    metatrain_folders = [os.path.join(metatrain_folder, label) \
                         for label in os.listdir(metatrain_folder) \
                         if os.path.isdir(os.path.join(metatrain_folder, label)) \
                         ]
    metatest_folders = [os.path.join(metatest_folder, label) \
                       for label in os.listdir(metatest_folder) \
                       if os.path.isdir(os.path.join(metatest_folder, label)) \
                       ]
    return metatrain_folders,metatest_folders

class CNNEncoder(nn.Module):
    """docstring for ClassName"""

    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
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

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(32 * 2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out

def main():
    ecci_client = Client()
    mqtt_thread = threading.Thread(target=ecci_client.initialize)
    mqtt_thread.start()
    ecci_client.wait_for_ready()

    FEATURE_DIM = args.feature_dim
    CLASS_NUM = args.class_num
    SAMPLE_NUM_PER_CLASS = 1
    BATCH_NUM_PER_CLASS = args.batch_num_per_class
    EPISODE = args.episode
    TEST_EPISODE = args.test_episode
    # LEARNING_RATE = args.learning_rate
    GPU = args.gpu
    # HIDDEN_UNIT = args.hidden_unit
    metatrain_character_folders, metatest_character_folders = omniglot_character_folders()

    for episode in range(EPISODE):
        data_msg_queue = ecci_client.get_sub_data_payload_queue()
        data_msg = data_msg_queue.get()
        feature_encoder = data_msg['feature_encoder']
        relation_network = data_msg['relation_network']

        if (episode == 0) or (episode == 1) or (episode % 10 == 0):
            print("Testing...")
            total_rewards = 0

            for i in range(TEST_EPISODE):
                task = tg.OmniglotTask(metatest_character_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,SAMPLE_NUM_PER_CLASS,)
                sample_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
                test_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="test",shuffle=True)

                sample_images,sample_labels = sample_dataloader.__iter__().next()
                test_images,test_labels = test_dataloader.__iter__().next()

                sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) # 5x64
                test_features = feature_encoder(Variable(test_images).cuda(GPU)) # 20x64
                sample_features_ext = sample_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
                test_features_ext = test_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
                test_features_ext = torch.transpose(test_features_ext,0,1)

                relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,5,5)
                relations = relation_network(relation_pairs).view(-1,CLASS_NUM)
                _,predict_labels = torch.max(relations.data,1)
                rewards = [1 if predict_labels[j]==test_labels[j].cuda(GPU) else 0 for j in range(CLASS_NUM)]
                total_rewards += np.sum(rewards)
            test_accuracy = total_rewards/1.0/CLASS_NUM/TEST_EPISODE
            print("episode:"+','+ str(episode)+','+"test accuracy:" + ',' + str(test_accuracy))

if __name__ == '__main__':
    main()

