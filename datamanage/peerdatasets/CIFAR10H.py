import os
import os.path
import sys
import pickle
import torch
import numpy as np
from .DatasetNumerical import DatasetNumerical
from utils.functions import check_integrity, download_and_extract_archive


class CIFAR10H(DatasetNumerical):
    """
    """
    base_folder = 'cifar-10-h'
    train_list = [
        'noise_21k.pkl',
        'noise_29k.pkl',
    ]

    test_list = [
        'test_batch',
    ]

    num_classes = 10

    def __init__(self, root=None, file_mark="21k", label_file_path=None, transform=None,
                 selected_idx=None, select_label=True, chosen_classes=None, classes_dist=None, is_mixup=False):
        assert file_mark in ["21k", "29k", "test"]
        self.file_mark = file_mark
        super(CIFAR10H, self).__init__(root=root, is_train=True, label_file_path=label_file_path, transform=transform, 
                         selected_idx=selected_idx, select_label=select_label, chosen_classes=chosen_classes, 
                         classes_dist=classes_dist, is_mixup=is_mixup)
        
    def load_data(self):

        self.data = []
        self.label = []

        # load from file
        if self.file_mark in ["21k", "29k"]:
            fname = f"noise_{self.file_mark}.pkl"
            fpath = os.path.join(self.root, self.base_folder, fname)
            with open(fpath, "rb") as f:
                data_tuple = pickle.load(f)
            
            self.data = data_tuple[0]
            self.label = data_tuple[1]

        elif self.file_mark == 'test':
            for file_name in self.test_list:
                file_path = os.path.join(self.root, self.base_folder, file_name)
                with open(file_path, 'rb') as f:
                    if sys.version_info[0] == 2:
                        entry = pickle.load(f)
                    else:
                        entry = pickle.load(f, encoding='latin1')
                    self.data.append(entry['data'])
                    if 'labels' in entry:
                        self.label.extend(entry['labels'])
                    else:
                        self.label.extend(entry['fine_labels'])

            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)

        self.data = torch.tensor( self.data, dtype=torch.float ) / 255.
        self.label = torch.tensor( self.label, dtype=torch.long )

        # self.true_label = self.label
        self.true_label = self.label.clone()

        # load label for noise label or co-training cases
        if (self.label_file_path is not None) and self.is_train:
            self.label = self.load_label()
        assert isinstance(self.label, torch.Tensor)
