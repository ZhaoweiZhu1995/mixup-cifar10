import os
import sys
import torch
import numpy as np
from .DatasetNumerical import DatasetNumerical

class View(DatasetNumerical):
    """
    """
    num_classes = 10

    def __init__(self, root=None, is_train=True, label_file_path=None, transform=None, 
                 selected_idx=None, select_label=True, chosen_classes=None, classes_dist=None):
        assert label_file_path is None
        super(View, self).__init__(root=root, is_train=is_train, label_file_path=label_file_path, transform=transform, 
                                   selected_idx=selected_idx, select_label=select_label,
                                   chosen_classes=chosen_classes, classes_dist=classes_dist)

    def load_data(self):
        '''
        _load_Views only works for Co-Training
        '''
        data_dict = torch.load(self.label_file_path)
        view_data = data_dict["view"]
        label = data_dict["label"]
        self.data = view_data
        self.label = label
        self.true_label = label.clone()