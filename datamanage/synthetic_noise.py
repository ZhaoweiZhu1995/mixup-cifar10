import os
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import random
from tqdm import tqdm
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from bisect import bisect_right

import matplotlib.pyplot as plt
import models
from models.peerreg import PeerRegModel
import datamanage.peerdatasets as datasets

import options
from utils.logger import Logger
from utils.functions import *
seed = 10086

if seed is not None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def pred_prob_dist(param_dict, model_log_path, model_name = None, low_bound=0., high_bound=1.0, bin_size=0.1, 
                   save_dir=None, fmark=None):
    # get input parameters
    # func 'get_parameter'; from utils
    opt = get_parameters(options.PeerOptions(), param_dict, None)
    bsize = opt.batch_size
    # setup device for running
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_idx
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"{'device:':>10} {device}")

    # setup data loader
    # prepare transforms
    train_trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4882, 0.4465], 
                                std =[0.2023, 0.1994, 0.2010])
    ])
    test_trans = transforms.Compose([
        transforms.Normalize(mean=[0.4914, 0.4882, 0.4465], 
                                std =[0.2023, 0.1994, 0.2010])
    ])

    # build datasets & setup dataloader
    data_root = "./datasets/"

    # noise labels setting
    dataset_path = os.path.join(data_root, "cifar-10-batches-py")
    path_list = [dataset_path, "noise_label", opt.noise_label_fname]
    label_file_path = os.path.join(*path_list) if opt.with_noise else None

    # validation setting; val_idx
    train_idx = None

    # Train & peer
    train_dataset = datasets.CIFAR10(root=data_root, is_train=True,
                                        transform=train_trans,
                                        label_file_path=label_file_path,
                                        selected_idx=train_idx,
                                        is_download=True)
    datasize = len(train_dataset)
    dataloader = DataLoader(dataset=train_dataset,
                                    batch_size=opt.batch_size,
                                    shuffle=False,
                                    num_workers=opt.num_workers)


    
    # loader = miniLoader(opt, is_train=True, is_shuffle=False) # load the training data
    # dataloader, datasize = loader.get_data()
    # dataroot = os.path.join(loader._dataset_path, "outlier_idx")

    # setup pretained model
    noisy_prior = None
    model = PeerRegModel(model_log_path, opt, noisy_prior)
    # model = NeuralNets(model_log_path, opt)
    model.load(model_name = model_name, is_predict=True)
    model.to(device)

    # segment the space according to the bin size
    # note if high_bound is not 1.0, the last bin is always [high_bound, 1.0)
    # thresholds = np.arange(low_bound, high_bound, step=bin_size)
    # num_count = np.zeros(len(thresholds)-1)
    length = np.floor(datasize/bsize).astype(int)
    prob_rec = np.zeros([length,bsize])
    prob_rec_2 = np.zeros([length,bsize])
    loss_rec_rec = np.zeros([length,bsize])


    # predict on training data and collect outlier index
    # idx_prob_label = [[raw_idx], [prob], [pred_label], [origin_label], [true_label]]
    idx_prob_label = torch.tensor([[], [], [], [], [], [], [], []], device=device)
    softmax = nn.Softmax(dim=-1)
    for i_batch, (inputs, labels, true_labels, raw_idx) in tqdm(enumerate(dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        true_labels = true_labels.to(device)
        # if (labels.cpu().numpy() != true_labels.cpu().numpy()).any():
        #     print(labels.cpu().numpy())
        #     print(true_labels.cpu().numpy())
        raw_idx = raw_idx.to(device)
        outputs, _ = model.predict(inputs)
        out_prob = softmax(outputs.detach())
        # probs, preds = torch.max(out_prob, 1)
        probs_top2, preds_top2 = out_prob.topk(max((1,2)), dim = 1)
        probs = probs_top2[:,0]
        preds = preds_top2[:,0]
        probs_2 = probs_top2[:,1]
        preds_2 = preds_top2[:,1]
        # print(out_prob[1,labels[1]])
        # exit()
        probs_label = torch.zeros_like(probs)
        for i in range(len(labels)):
            probs_label[i] = out_prob[i,labels[i]]
        # probs_label = [out_prob[i,labels[i]][0] for i in range(len(labels))]
        # print(probs_label)
        # exit()
        # print(out_prob.shape)
        # print(preds.shape)
        # loss_rec = torch.log(probs+1e-8)  + (torch.log(probs+1e-8) - torch.sum(torch.log(out_prob+1e-8), dim=1))/9
        loss_rec = torch.log(probs_label+1e-8)  - torch.mean(torch.log(out_prob+1e-8), dim=1)
        # loss_rec = torch.log(probs_label+1e-8)
        # print(loss_rec.shape)
#   - (torch.log(out_prob[:,preds]+1e-8) - torch.sum(out_prob, dim=1))/9
        # exit()
        
        

        batch_record = torch.cat((raw_idx.float().view(1,-1), 
                                  probs.view(1,-1),
                                  probs_2.view(1,-1), 
                                  preds.float().view(1,-1),
                                  preds_2.float().view(1,-1), 
                                  labels.float().view(1,-1), 
                                  true_labels.float().view(1,-1),
                                  loss_rec.view(1,-1)), dim=0)
        idx_prob_label = torch.cat((idx_prob_label, batch_record), dim=1)

        
        if i_batch<length:
            prob_rec[i_batch] = probs.cpu().numpy()
            prob_rec_2[i_batch] = probs_2.cpu().numpy()
            loss_rec_rec[i_batch] = loss_rec.cpu().numpy()


        # for idx in range(labels.shape[0]):
        #     prob_idx = bisect_right(thresholds, probs[idx].item()) - 1
        #     num_count[prob_rec[idx]] += 1

    prob_rec = np.append(prob_rec.reshape(-1),probs.cpu().numpy())
    prob_rec_2 = np.append(prob_rec_2.reshape(-1),probs_2.cpu().numpy())
    loss_rec_rec = np.append(loss_rec_rec.reshape(-1),loss_rec.cpu().numpy())
    print(max(loss_rec_rec))
    print(min(loss_rec_rec))
    thresholds = np.arange(min(loss_rec_rec), max(loss_rec_rec), step=bin_size)
    idx_prob_label_np = idx_prob_label.cpu().numpy()
    print(f'thresholds are {thresholds}')

    num_count, _ = np.histogram(prob_rec, thresholds)
    num_count = num_count/sum(num_count)
    print(f'num_count is {num_count}')
    num_count_2, _ = np.histogram(prob_rec_2, thresholds)
    num_count_2 = num_count_2/sum(num_count_2)
    print(f'num_count is {num_count_2}')

    # exit()
    # save data for further use
    fname = "idx_prob_label.pkl" if fmark is None else f"idx_prob_label_{fmark}.pkl"
    save_data(idx_prob_label_np, save_dir, fname)

    fname = "num_count.pkl" if fmark is None else f"num_count_{fmark}.pkl"
    save_data(num_count, save_dir, fname)

    return thresholds, num_count, num_count_2, idx_prob_label_np, loss_rec_rec

def get_confusion_matrix(param_dict, preds, true_labels):
    opt = get_parameters(options.PeerOptions(), param_dict, None)
    conf_mat = np.zeros((opt.num_classes,opt.num_classes))
    for i in range(len(preds)):
        # print(f"true_labels[i]:{true_labels[i]}, preds[i]:{preds[i]}")
        conf_mat[true_labels[i]][preds[i]] += 1
    return conf_mat/preds.shape[0]*opt.num_classes

def save_data(data, save_dir, fname):
    file_path = os.path.join(save_dir, fname)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"data {fname} saved to {save_dir}")

def plot_prob_dist2(name, thresholds, num_count_c, num_count_n, figure_title = 'Loss distribution'):
    fig = plt.figure(figsize=(7,5))
    ax  = fig.add_subplot(1,1,1)

    # ax.plot(thresholds[1:], num_count_c, marker='.', label="clean")
    # ax.plot(thresholds[1:], num_count_n, marker='.', label="noisy")
    ax.hist(num_count_c, bins = thresholds[1:], alpha = 0.5, label = "clean")
    ax.hist(num_count_n, bins = thresholds[1:], alpha = 0.5, label = "noisy")

    ax.set_xlabel('Loss', fontsize = 19)
    # ax.set_xlim(0., 1.0)
    ax.set_ylabel("Counting", fontsize = 19)
    ax.set_title(figure_title)
    ax.legend()

    fig.savefig(f"./{name}.png", dpi=200)
    plt.close(fig)

# NOTE we name the following function as load_idx_by_prob, not because
# the idx are loaded according to their predcited probabilities,
# just for matching the name of the previous method: get_idx_by_prob
# def load_idx_by_prob(data_path, thresholds):
#     # prepare data path
#     if data_path is None:
#         data_path = "datasets/CIFAR/cifar-10-batches-py/outlier_idx/outlier_idx.pt"

#     # prepare keys
#     keys = {}
#     upper = 1.0
#     for val in thresholds:
#         key_str = f"({str(upper)},{str(val)}]"
#         keys[val] = key_str
#         upper = val

#     # load data
#     data = torch.load(data_path)

#     # check integrity
#     for val in thresholds:
#         # print(keys[val])
#         # for i in range(10):
#         #     print(f"num for class {i}: {torch.sum(data[keys[val]][1] == i)}")
#         if keys[val] not in data.keys():
#             raise ValueError(f"not valid key {keys[val]}")
#     assert len(data) == len(thresholds), "data and thresholds not match "

#     return keys, data

def get_idx(thre, probs, preds, true_labels, save_dir, sel_idx, is_save = True, fname = None, fmark = None, loss_rec = None, thresholds = None, labels = None):
    clean_idx = []
    noisy_idx = []
    raw_idx = []
    raw_label = []
    clean_set = []
    noisy_label = []
    loss_c = []
    loss_n = []
    print(loss_rec.shape)
    for i in range(len(probs)):
        if labels[i] == true_labels[i] or i in sel_idx:
            loss_c += [loss_rec[i]]
            # exit()
        else:
            loss_n += [loss_rec[i]]

        flag = probs[i] > thre
        noisy_label += [preds[i]]
        if flag or i in sel_idx:
            clean_idx += [i]
            if i not in sel_idx:
                clean_set += [i]
            else:
                noisy_label[-1] = true_labels[i]
        else:
            noisy_idx += [i]
        raw_idx += [i]
        raw_label += [true_labels[i]]
        # if i == 10:
        #     print(clean_idx)
        #     print(raw_label)
        #     exit()

    # print(loss_c.shape)
    # print(loss_n.shape)
    # exit()
    # num_count_c, _ = np.histogram(loss_c, thresholds)
    # num_count_n, _ = np.histogram(loss_n, thresholds)
    
    # print(f'Number of corrects in clean_idx: {num_corr_clean}, noise rates: {num_corr_clean/len(clean_idx)}')
    # print(f'Number of corrects in noisy_idx: {num_corr_noisy}, noise rates: {num_corr_noisy/len(noisy_idx)}')

    dict_save = { 'clean_idx':  clean_idx, 'noisy_idx':  noisy_idx, 'raw_idx':  raw_idx, 'raw_label': raw_label, 'noisy_label': noisy_label}
    
    # if fname is None:
    #     fname = "noise_label_train.pt"

    # # save noise labels
    # fpath = 'datasets/cifar-10-batches-py/noise_label/r5_noise.pt'
    # noise_labels = np.array(noisy_label)
    # original_labels = np.array(raw_label)
    # if isinstance(noise_labels, np.ndarray):
    #     torch.save({"noise_label_train": torch.from_numpy(noise_labels).long(),
    #                 "clean_label_train": torch.from_numpy(original_labels).long()}, fpath)
    
    # fmark = f'clean_noisy_idx'
    # if is_save:
    #     if fname is None:
    #         fname = "C10_train_r5"
    #         fname += f"_{fmark}.pkl" if fmark is not None else ".pkl"
    #         save_data(dict_save, save_dir, fname)
    # else:
    #     return dict_save
    return loss_c, loss_n


if __name__ == "__main__":
    # set dataset root
    # data_root = "datasets/cifar-10-batches-py/outlier_idx" # for CIFAR10 outlier idx
    # model_log_path = "datasets/cifar-10-batches-py/trained_models/CE_cnn_13_4000_2020-02-21-11-25"
    # model_log_path = "/root/zhuzhw/March/PeerLearning/logs/CIFAR10/resnet_cifar18_pre/CE_CE_4k_2020-03-14-07-35"
    # model_log_path = "/root/zhuzhw/March/PeerLearning/logs/CIFAR10_before317/resnet_cifar18_pre/CE_CE_r5_2020-03-16-06-35"
    # model_log_path = "/root/zhuzhw/March/PeerLearning/logs/CIFAR10/resnet_cifar18_pre/CE_peer_pred_2020-03-20-07-36"
    # model_log_path = "/root/zhuzhw/March/PeerLearning/logs/CIFAR10/resnet_cifar18_pre/CE_CE_pred_2020-03-20-07-52"
    # model_log_path = "/root/zhuzhw/March/PeerLearning/logs/CIFAR10/resnet_cifar18_pre/CE_CE_pred_2020-03-20-03-03"
    model_log_path = "/root/zhuzhw/March/PeerLearning/logs/CIFAR10/resnet_cifar18_pre/CE_peer_pred_2020-03-20-03-02"
    # model_log_path = "/root/zhuzhw/March/PeerLearning/logs/CIFAR10_before317/resnet_cifar18_pre/CE_CE_r5_2020-03-16-06-35/"
    # model_log_path = "/root/zhuzhw/March/PeerLearning/logs/CIFAR10_before317/resnet_cifar18_pre/CE_peer_r5_correctprior_2020-03-17-03-46/"
    # 10 epochs
    # model_log_path = "/root/zhuzhw/March/PeerLearning/logs/CIFAR10/resnet_cifar18_pre/CE_CE_pred_2020-03-20-09-55/"
    # model_log_path = "/root/zhuzhw/March/PeerLearning/logs/CIFAR10/resnet_cifar18_pre/CE_peer_pred_2020-03-20-09-55/"
    noise_file = "noise_label_train_random_flip_r0_5_CIFAR10.pt"
    figure_title = 'peer pred 200 epoch model + r5 noise'
    # noise_file = "prediction_noise_new.pt"
    # bin settings;
    # note if high_bound is not 1.0, the last bin is always [high_bound, 1.0)
    low_bound = 0 # 0
    high_bound = 7  # 1.01
    bin_size = 0.3
    thre = 0.7
    save_dir = "./"
    fmark = "r5"
    # data_path = os.path.join(data_root, save_name)
    param_dict = {"--is_train": True,
                        "--is_plot_results": True,
                        "--exp_name": "CIFAR10_prob_dist",
                        "--dataset": "CIFAR10",
                        "--netARCH": "resnet_cifar18_pre",
                        "--num_classes": 10,
                        "--batch_size": 256,
                        "--optimizer": "SGD",
                        "--is_validate": False,
                        "--with_noise": True,
                        "--is_peerloss": False,
                        "--gpu_idx": '0',
                        "--noise_label_fname": noise_file,
                       }
    thresholds, num_count, num_count_2, idx_prob_label, loss_rec = \
        pred_prob_dist(param_dict = param_dict,
                       model_log_path = model_log_path,
                       model_name = 'model_present_best_val.pt',   # cotraining_model_cf0_step_0
                       low_bound = low_bound, 
                       high_bound = high_bound,
                       bin_size = bin_size,
                       save_dir = save_dir,
                       fmark = fmark
                      )


    idx_prob_label = pickle.load( open( "idx_prob_label_r5.pkl", "rb" ) )
    # idx_prob_label = pickle.load( open( "cnn13_train_4000_clean_noisy_idx.pkl", "rb" ) )
    # raw_idx = idx_prob_label[0]
    probs = idx_prob_label[1]
    # probs_2 = idx_prob_label[2]
    preds = idx_prob_label[3].astype(int)
    # preds_2 = idx_prob_label[4].astype(int)
    labels = idx_prob_label[5].astype(int)


    true_labels = idx_prob_label[6].astype(int)
    print(f'Number of corrects: {sum(preds==true_labels)}')
    print(f'Ratio of corrects: {sum(preds==true_labels)/50000}')
    print(f'Check: {sum(true_labels==labels)}')
    
    
    loss_c, loss_n = get_idx(thre = thre, probs = probs, preds = preds, true_labels = true_labels, save_dir = save_dir, sel_idx = [], fmark = fmark, loss_rec = loss_rec, thresholds = thresholds, labels = labels)

    # dict_save = pickle.load( open( "cnn13_train_4000_clean_noisy_idx.pkl", "rb" ) )
    # tmp = dict_save['raw_label'][0:100]
    # print(f'raw_train label 2 is: {tmp}')
    # tmp = dict_save['raw_label'][0:4000]
    # print(f'check sum-2:{sum(tmp==labels[:4000])}')
    # conf_mat = get_confusion_matrix(param_dict=param_dict, preds = noisy_label, true_labels = true_labels)

    # # conf_mat_2 = get_confusion_matrix(param_dict=param_dict, 
    # #                                 preds = idx_prob_label[4].astype(int), 
    # #                                 true_labels = idx_prob_label[6].astype(int)
    # #                                 )

    # print(np.around(conf_mat,decimals = 2))
    
    # plot_prob_dist(name = 'r5distfig_1', thresholds = thresholds, num_count = num_count)
    # plot_prob_dist(name = 'r5distfig_2', thresholds = thresholds, num_count = num_count_2)
    # print(num_count_c)
    # print(num_count_n)
    plot_prob_dist2(name = 'loss', thresholds = thresholds, num_count_c = loss_c, num_count_n = loss_n, figure_title = figure_title)



    # keys, idx_dict = load_idx_by_prob(data_path, thresholds)
    # for val in thresholds:
    #     print(keys[val])
    #     for i in range(10):
    #         print(f"num for class {i}: {torch.sum(idx_dict[keys[val]][1] == i)}")
