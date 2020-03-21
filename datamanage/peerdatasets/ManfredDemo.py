import os
import os.path
import sys
import torch
import numpy as np
from .DatasetNumerical import DatasetNumerical


class ManfredDemo(DatasetNumerical):
    """
    """
    num_classes = 2
    
    def __init__(self, root=None, is_train=True, label_file_path=None, transform=None, 
                       selected_idx=None, select_label=True, chosen_classes=None, classes_dist=None,
                       noise_type=None, noise_level=None):
        self.noise_type = noise_type
        self.noise_level = noise_level
        super(ManfredDemo, self).__init__(root=root, is_train=is_train, label_file_path=label_file_path, transform=transform,
                                          selected_idx=selected_idx, select_label=select_label, 
                                          chosen_classes=chosen_classes, classes_dist=classes_dist)

    def load_data(self):
        if self.noise_type is not None:
            print("Manfred Demo with noise")
            assert self.noise_level is not None

        width = 256.
        height = 256.
        outerRingRadiusStart = 0.33
        outerRingRadiusEnd = 0.48
        innerRingRadiusEnd = 0.2
        deltaRadius = 0.03
        numTrainingPoints = 1000

        data = np.ndarray(shape=(numTrainingPoints, 2))
        label = np.ndarray(shape=(numTrainingPoints,))
        true_label = np.ndarray(shape=(numTrainingPoints,))
        radius = np.ndarray(shape=(numTrainingPoints,))
        # generate data
        for i in range(numTrainingPoints):
            if i % 2 == 0:
                radius[i] = np.random.uniform(width * outerRingRadiusStart, width * outerRingRadiusEnd)
                angle = np.random.uniform(0, 2 * 3.147)
                data[i,0] = np.round(width * 0.5 + radius[i] * np.cos(angle))
                data[i,1] = np.round(height * 0.5 + radius[i] * np.sin(angle))
                label[i] = 1
                true_label[i] = 1
                if (self.noise_type is not None) and self.is_train:
                    rand = np.random.uniform()
                    if self.noise_type == "lowmargin":
                        if radius[i] < (outerRingRadiusStart + deltaRadius)* width:
                            if rand < self.noise_level:
                                label[i] = 0
                    elif self.noise_type == "highmargin":
                        if radius[i] > (outerRingRadiusEnd - deltaRadius) * width:
                            if rand < self.noise_level:
                                label[i] = 0
                    elif self.noise_type == "random":
                        if rand < self.noise_level:
                            label[i] = 0
            else:
                radius[i] = np.random.uniform(0, width * innerRingRadiusEnd)
                angle = np.random.uniform(0, 2 * 3.147)
                data[i,0] = np.round(width * 0.5 + radius[i] * np.cos(angle))
                data[i,1] = np.round(height * 0.5 + radius[i] * np.sin(angle))
                label[i] = 0
                true_label[i] = 0
                if (self.noise_type is not None) and self.is_train:
                    rand = np.random.uniform()
                    if self.noise_type == "lowmargin":
                        if radius[i] > (innerRingRadiusEnd - deltaRadius) * width:
                            if rand < self.noise_level:
                                label[i] = 1
                    elif self.noise_type == "highmargin":
                        if radius[i]< (deltaRadius) * width:
                            if rand < self.noise_level:
                                label[i] = 1
                    elif self.noise_type == "random":
                        if rand < self.noise_level:
                            label[i] = 1

        self.data = torch.from_numpy(data).float()
        self.label = torch.from_numpy(label).long() # note we use long type not float
        self.true_label = torch.from_numpy(true_label).long()

