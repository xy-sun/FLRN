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
# import task_generator as tg
import os
import math
import argparse
# import scipy as sp
# import scipy.stats
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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

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
        # out = out.view(out.size(0),-1)
        return out  # 64

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(32 * 2, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size * 3 * 3, hidden_size)
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
    # Initialize ecci sdk and connect to the broker in central-cloud
    # ecci_client.initialize()
    mqtt_thread = threading.Thread(target=ecci_client.initialize)
    mqtt_thread.start()
    # Wait for the container on the side to be ready
    ecci_client.wait_for_ready()

    # Hyper Parameters
    FEATURE_DIM = args.feature_dim
    RELATION_DIM = args.relation_dim
    CLASS_NUM = args.class_num
    SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
    BATCH_NUM_PER_CLASS = args.batch_num_per_class
    EPISODE = args.episode
    TEST_EPISODE = args.test_episode
    LEARNING_RATE = args.learning_rate
    GPU = args.gpu
    HIDDEN_UNIT = args.hidden_unit

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(FEATURE_DIM, RELATION_DIM)

    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=100000, gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=100000, gamma=0.5)

    for episode in range(EPISODE):
        ecci_client.send_message(payload, ["sur_edge1", "sur_edge2", "sur_edge3", "sur_edge4", "sur_edge5", "sur_test"])
        payload = {"feature_encoder": feature_encoder}
        ecci_client.send_message(payload, ["client1", "client2", "client3", "client4", "client5"])
        payload = {"feature_encoder": feature_encoder, "relation_network": relation_network}
        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)
        data_msg_queue = ecci_client.get_sub_data_payload_queue()
        loss_list = []
        for i in range(5):
            data_msg = data_msg_queue.get()
            batch_labels = data_msg['batch_labels']
            sample_features = data_msg['sample_features']
            batch_features = data_msg['batch_features']
            sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
            batch_features_ext = batch_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
            batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
            relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, FEATURE_DIM * 2, 19, 19)
            relations = relation_network(relation_pairs).view(-1, CLASS_NUM)
            mse = nn.MSELoss().cuda(GPU)
            one_hot_labels = Variable(
                torch.zeros(BATCH_NUM_PER_CLASS * CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1, 1), 1)).cuda(
                GPU)
            loss = mse(relations, one_hot_labels)
            loss_list.append(loss)
        loss = (loss_list[0] + loss_list[1] + loss_list[2] + loss_list[3] + loss_list[4]) / 5
        feature_encoder.zero_grad()
        relation_network.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(), 0.5)
        feature_encoder_optim.step()
        relation_network_optim.step()

if __name__ == '__main__':
    main()
