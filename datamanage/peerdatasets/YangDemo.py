import os
import os.path
import sys
import torch
import numpy as np
from .DatasetNumerical import DatasetNumerical


class YangDemo(DatasetNumerical):
    """
    """
    num_classes = 2
    
    def __init__(self, root=None, is_train=True, label_file_path=None, transform=None, 
                       selected_idx=None, select_label=True, chosen_classes=None, classes_dist=None,
                       noise_type=None, noise_level=None):
        self.noise_type = noise_type
        self.noise_level = noise_level
        super(YangDemo, self).__init__(root=root, is_train=is_train, label_file_path=label_file_path, transform=transform,
                                          selected_idx=selected_idx, select_label=select_label, 
                                          chosen_classes=chosen_classes, classes_dist=classes_dist)
    def generate_data_in_circle(self,center,radius,num,label):
        num = round(num)
        data = np.zeros([num,3])
        radius_sample = np.random.uniform(low = 0, high = radius, size = num)
        angle_sample = np.random.uniform(low = 0, high = 2*np.pi, size = num)
        data[:,0] = center[0] + radius_sample * np.cos(angle_sample)
        data[:,1] = center[1] + radius_sample * np.sin(angle_sample)
        data[:,2] = label
        return data

    def load_data(self):
        if self.noise_type is not None:
            print("Yang Demo with noise")
            assert self.noise_level is not None

        width = 256.
        height = 256.

        numTrainingPoints = 500

        ratio = [0.2,0.6,0.2] # LM, Pe, Pu

        center = 128
        leftLargeMarginCenter = np.array([25/1.73, center])
        rightLargeMarginCenter = 256-leftLargeMarginCenter
        LargeMarginRadius = 25/1.73

        leftPenalizerCenter = np.array([center-30, center+30]) 
        rightPenalizerCenter = 256 - leftPenalizerCenter
        PenalizerRadius = 25  
        
        leftPullerCenter = np.array([center-30, 20])
        rightPullerCenter = 256 - leftPullerCenter
        PullerRadius = 25/1.73

        
        dataLM0 = self.generate_data_in_circle(leftLargeMarginCenter,LargeMarginRadius, num = numTrainingPoints*ratio[0],label=0)
        dataPe0 = self.generate_data_in_circle(leftPenalizerCenter,PenalizerRadius, num = numTrainingPoints*ratio[1],label=0)
        dataPu0 = self.generate_data_in_circle(leftPullerCenter,PullerRadius, num = numTrainingPoints*ratio[2],label=0)
  
        dataLM1 = self.generate_data_in_circle(rightLargeMarginCenter,LargeMarginRadius, num = numTrainingPoints*ratio[0],label=1)
        dataPe1 = self.generate_data_in_circle(rightPenalizerCenter,PenalizerRadius, num = numTrainingPoints*ratio[1],label=1)
        dataPu1 = self.generate_data_in_circle(rightPullerCenter,PullerRadius, num = numTrainingPoints*ratio[2],label=1)
        
        if 'PE' in self.noise_type:
            dataPe0,dataPe1 = self.add_noise(dataPe0,dataPe1,self.noise_level)
        if 'LM' in self.noise_type:
            dataLM0,dataLM1 = self.add_noise(dataLM0,dataLM1,self.noise_level)
        if 'PU' in self.noise_type:
            dataPu0,dataPu1 = self.add_noise(dataPu0,dataPu1,self.noise_level)
        data0 = np.concatenate((dataLM0,dataPe0,dataPu0))
        data1 = np.concatenate((dataLM1,dataPe1,dataPu1))
        # plt.scatter(data0[:,0], data0[:,1], color='red', marker = '.')
        # plt.scatter(data1[:,0], data1[:,1], color='blue', marker = '.')
        # plt.plot([-10,256+10],[center,center], linestyle='--', c = '0.75')

        # data0[:,-1] = np.random.choice(2, numTrainingPoints, p = [1-self.noise_level, self.noise_level])
        # data1[:,-1] = np.random.choice(2, numTrainingPoints, p = [self.noise_level, 1-self.noise_level])
        data = np.concatenate((data0,data1))
        np.random.shuffle(data)
        self.label = torch.from_numpy(data[:,-1]).long()
        self.data = torch.from_numpy(data[:,:-1]).float()
        self.true_label = self.label.clone()
    def add_noise(self,data0,data1,noise_level):
        num = data0.shape[0]
        data0[:,-1] = np.random.choice(2, num, p = [1-noise_level[0], noise_level[0]])
        data1[:,-1] = np.random.choice(2, num, p = [noise_level[1], 1-noise_level[1]])
        return data0,data1



        # self.data = torch.from_numpy(data).float()
        # self.label = torch.from_numpy(label).long() # note we use long type not float
        # self.true_label = torch.from_numpy(label).long()

