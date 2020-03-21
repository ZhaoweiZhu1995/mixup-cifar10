import os
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from tqdm import tqdm
import pickle
import numpy as np
import torch
import torch.nn as nn

import models
from datamanage.dataloader import miniLoader
import options
from utils.logger import Logger
from utils.functions import *


def get_idx_by_prob(param_dict, model_log_path, thresholds, save_name=None):
    # get input parameters
    # func 'get_parameter'; from utils
    opt = get_parameters(options.PeerOptions(), param_dict, None)

    # setup device for running
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_idx
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"{'device:':>10} {device}")

    # setup data loader
    loader = miniLoader(opt, is_train=True, is_shuffle=False) # load the training data
    dataloader, datasize = loader.get_data()
    dataroot = os.path.join(loader.dataset_path, "outlier_idx")

    # setup pretained model
    model = models.BasePeerModel(model_log_path, opt)
    model.load(is_predict=True)
    model.to(device)

    # init outlier index dict
    outlier_idx_dict = {}
    keys = {}
    upper = 1.0
    for val in thresholds:
        key_str = f"({str(upper)},{str(val)}]"
        keys[val] = key_str
        outlier_idx_dict[key_str] = torch.tensor([[],[]]).long() #record raw_idx & label
        upper = val

    # predict on training data and collect outlier index
    softmax = nn.Softmax(dim=-1)
    for i_batch, (inputs, labels, true_labels, raw_idx) in tqdm(enumerate(dataloader)):
        inputs = inputs.to(device)
        outputs, _ = model.predict(inputs)
        out_prob = softmax(outputs.detach())
        probs, preds = torch.max(out_prob, 1)

        for idx in range(labels.shape[0]):
            if preds[idx] == labels[idx]:
                upper = 1.0
                for val in thresholds:
                    if probs[idx] >= val and probs[idx] < upper:
                        new_ele = torch.tensor([[raw_idx[idx]], [labels[idx]]])
                        outlier_idx_dict[ keys[val] ] = torch.cat((outlier_idx_dict[ keys[val] ], new_ele), dim=1)
                    upper = val

    # save for further use
    for val in thresholds:
        print(outlier_idx_dict[ keys[val] ].shape)
    if save_name is None:
        save_path = os.path.join(dataroot, "outlier_idx.pt")
    else:
        save_path = os.path.join(dataroot, save_name)
    torch.save(outlier_idx_dict, save_path)
    print(f"data saved to {save_path}")

# NOTE we name the following function as load_idx_by_prob, not because
# the idx are loaded according to their predcited probabilities,
# just for matching the name of the previous method: get_idx_by_prob
def load_idx_by_prob(data_path, thresholds):
    # prepare data path
    if data_path is None:
        data_path = "datasets/CIFAR/cifar-10-batches-py/outlier_idx/outlier_idx.pt"

    # prepare keys
    keys = {}
    upper = 1.0
    for val in thresholds:
        key_str = f"({str(upper)},{str(val)}]"
        keys[val] = key_str
        upper = val

    # load data
    data = torch.load(data_path)

    # check integrity
    for val in thresholds:
        # print(keys[val])
        # for i in range(10):
        #     print(f"num for class {i}: {torch.sum(data[keys[val]][1] == i)}")
        if keys[val] not in data.keys():
            raise ValueError(f"not valid key {keys[val]}")
    assert len(data) == len(thresholds), "data and thresholds not match "

    return keys, data


# Distribution of stable labels
# Calculate the matrix between class and label
def distribution_matrix(label_noisy, label_real):
    category_noisy = np.unique(label_noisy)
    category_real = np.unique(label_real)
    row = len(category_noisy)
    column = len(category_real)
    s = (row, column)
    result = np.zeros(s)
    for i in range(row):
        for j in range(column):
            result[i, j] = np.count((label_noisy == category_noisy[i]) | (label_real == category_real[j]))



if __name__ == "__main__":
    # set dataset root
    data_root = "datasets/cifar-10-batches-py/outlier_idx" # for CIFAR10 outlier idx
    model_log_path = "datasets/cifar-10-batches-py/ce_clean_base_2020-01-06-10-31"

    # set thresholds
    # thresholds = [0.9999999, 0.999999]
    thresholds = [0.99999]

    thresholds_str = str(thresholds[-1]).replace(".", "_")
    save_name = f"outlier_idx_threshold-{thresholds_str}.pt"
    data_path = os.path.join(data_root, save_name)

    get_idx_by_prob({"--is_train": True,
                        "--is_plot_results": True,
                        "--exp_name": "CIFAR10_outlier_idx",
                        "--dataset": "CIFAR10",
                        "--ARCH": "resnet_cifar18",
                        "--num_classes": 10,
                        "--batch_size": 256,
                        "--is_balanced": False,
                        "--is_normalize": True,
                        "--is_validate": False,
                        "--with_noise": False,
                        "--is_peerloss": False,
                        "--gpu_idx": '2'
                    },
                    model_log_path = model_log_path,
                    thresholds = thresholds,
                    save_name = save_name
                )

    keys, idx_dict = load_idx_by_prob(data_path, thresholds)
    for val in thresholds:
        print(keys[val])
        for i in range(10):
            print(f"num for class {i}: {torch.sum(idx_dict[keys[val]][1] == i)}")
