import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datamanage import peerdatasets as datasets

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt



class miniLoader(object):
    def __init__(self, opt, is_train=True, is_val=False, is_shuffle=True, view_idx=None, is_mixup=False):
        self._name = "miniloader for PeerLearning"

        path_dict = {
            "CIFAR10": ["cifar-10-batches-py"],
            "CIFAR100": ["cifar-100-python"],
            "MNIST": ["MNIST"],
            "FashionMNIST": ["FashionMNIST"],
            "ManfredDemo": ["no_need"],
            "YangDemo": ["no_need"],
        }
        path_list = [opt.data_root] + path_dict[opt.dataset]
        self._dataset_path = os.path.join( *path_list )
        self._opt = opt
        self._is_train = is_train
        self._is_val = is_val
        self._is_shuffle = is_shuffle
        self._view_idx = view_idx # for using ECCV view in cotraining only
        self._is_mixup = is_mixup

    @property
    def dataset_path(self):
        return self._dataset_path

    def get_data(self, data_index=None, select_label=True, label_file_path=None,
                       chosen_classes=None, classes_dist=None):
        '''
        '''
        cfg_dict = {}
        cfg_dict["root"] = self._opt.data_root
        cfg_dict["is_train"] = self._is_train
        cfg_dict["selected_idx"] = data_index
        cfg_dict["select_label"] = select_label
        cfg_dict["chosen_classes"] = chosen_classes
        cfg_dict["classes_dist"] = classes_dist

        if self._opt.dataset == "CIFAR10":
            # trans adopted from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
            trans_list = []
            if self._is_train and (not self._is_val):
                trans_list += [transforms.ToPILImage(),
                               transforms.RandomCrop(32, 4),
                               transforms.RandomHorizontalFlip(), 
                               transforms.ToTensor()]
            # trans_list += [transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            #                                     std=[0.229, 0.224, 0.225])]
            trans_list += [transforms.Normalize(mean=[0.4914, 0.4882, 0.4465], 
                                                std =[0.2023, 0.1994, 0.2010])]
            trans = transforms.Compose(trans_list)

            cfg_dict["is_download"] = True
            cfg_dict["label_file_path"] = label_file_path
            cfg_dict["transform"] = trans
            cfg_dict["is_mixup"] = self._is_mixup

            self._dataset = datasets.CIFAR10( **cfg_dict )
        
        elif self._opt.dataset == "CIFAR100":
            # trans adopted from https://github.com/meliketoy/wide-resnet.pytorch/blob/master/
            trans_list = []
            if self._is_train and (not self._is_val):
                trans_list += [transforms.ToPILImage(),
                               transforms.RandomCrop(32, 4),
                               transforms.RandomHorizontalFlip(), 
                               transforms.RandomRotation(15),
                               transforms.ToTensor()]
            trans_list += [transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                                                std =[0.2675, 0.2565, 0.2761])]
            trans = transforms.Compose(trans_list)

            cfg_dict["is_download"] = True
            cfg_dict["label_file_path"] = label_file_path
            cfg_dict["transform"] = trans
            cfg_dict["is_mixup"] = self._is_mixup

            self._dataset = datasets.CIFAR100( **cfg_dict )
        
        elif self._opt.dataset == "SVHN":
            # trans follows the Deep Co-Training setting
            # https://github.com/AlanChou/Deep-Co-Training-for-Semi-Supervised-Image-Recognition/blob/master/main.py
            trans_list = []
            if self._is_train and (not self._is_val):
                trans_list += [
                    transforms.ToPILImage(),
                    transforms.RandomAffine(0, translate=(1/16,1/16)), # translation at most two pixels
                    transforms.ToTensor()
                ]
            # trans_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            trans = transforms.Compose(trans_list)

            cfg_dict["is_download"] = True
            cfg_dict["label_file_path"] = label_file_path
            cfg_dict["transform"] = trans

            self._dataset = datasets.SVHN( **cfg_dict )

        elif self._opt.dataset == "MNIST":
            # trans adopted from https://github.com/pytorch/examples/blob/master/mnist/main.py
            # trans_list = []
            # trans_list += [transforms.Normalize((0.1307,), (0.3081,))]
            # trans = transforms.Compose(trans_list)
            trans = None
            
            cfg_dict["is_download"] = True
            cfg_dict["label_file_path"] = label_file_path
            cfg_dict["transform"] = trans

            self._dataset = datasets.MNIST( **cfg_dict )
            
        elif self._opt.dataset == "FashionMNIST":
            trans = None
            
            cfg_dict["is_download"] = True
            cfg_dict["label_file_path"] = label_file_path
            cfg_dict["transform"] = trans
            
            self._dataset = datasets.FashionMNIST( **cfg_dict )
        
        elif self._opt.dataset == "ManfredDemo":
            # we will never load label file for ManfredDemo case
            cfg_dict["label_file_path"] = None
            cfg_dict["transform"] = None
            dnt = self._opt.demo_noise_type
            cfg_dict["noise_type"] = dnt if dnt != "none" else None
            cfg_dict["noise_level"] = self._opt.demo_noise_level

            self._dataset = datasets.ManfredDemo( **cfg_dict )

        elif self._opt.dataset == "YangDemo":
            # we will never load label file for ManfredDemo case
            cfg_dict["label_file_path"] = None
            cfg_dict["transform"] = None
            dnt = self._opt.Ydemo_noise_type
            cfg_dict["noise_type"] = dnt if dnt != "none" else None
            cfg_dict["noise_level"] = self._opt.Ydemo_noise_level

            self._dataset = datasets.YangDemo( **cfg_dict )
        
        elif self._opt.dataset == "ECCV_CIFAR10_views":
            # we load from fixed path 
            train_data_name = f"CIFAR10_train_view_{self._view_idx+1}_best4000.pt"
            test_data_name = f"CIFAR10_test_view_{self._view_idx+1}_best4000.pt"
            if self._is_train:
                fpath = os.path.join(self._dataset_path, train_data_name)
            else:
                fpath = os.path.join(self._dataset_path, test_data_name)

            cfg_dict["label_file_path"] = fpath
            cfg_dict["transform"] = None
            
            self._dataset = datasets.View( **cfg_dict )

        else:
            raise ValueError("invalid dataset")

        dataloader = DataLoader(self._dataset, 
                                batch_size = self._opt.batch_size, 
                                shuffle = self._is_shuffle, 
                                num_workers = self._opt.num_workers,
                                drop_last=False)
        datasize = len(self._dataset)

        return dataloader, datasize

    def get_peer(self, bsize, peer_pool=None):
        '''
        presently, adopt uniform sampling;
        and account for the number of samples in a minibatch
        ----
        peer_pool: a list of indices from which peer samples are drawn
        '''
        if peer_pool is None:
            datasize = len(self._dataset)
        else:
            datasize = peer_pool
        idx_x = np.random.choice(datasize, size = bsize * self._opt.peer_size)
        idx_y = np.random.choice(datasize, size = bsize * self._opt.peer_size)

        if self._is_mixup:
            (x_peer_1, x_peer_2, _, _, _) = self._dataset[idx_x]
            (_, _, y_peer, _, _) = self._dataset[idx_y]

            return x_peer_1, x_peer_2, y_peer
        else:
            (x_peer, _, _, _) = self._dataset[idx_x]
            (_, y_peer, _, _) = self._dataset[idx_y]

            return x_peer, y_peer
    
    def get_raw_data(self):
        return self._dataset.get_raw_data()



if __name__ == "__main__":
    C = type("options", (object,), {})
    opt = C()
    setattr(opt, "data_root", "./datasets")
    setattr(opt, "dataset", "CIFAR10")
    setattr(opt, "batch_size", 1)
    setattr(opt, "shuffle", False)
    setattr(opt, "num_workers", 0)
    setattr(opt, "num_classes", 10)
    setattr(opt, "is_normalize", True)
    setattr(opt, "is_balanced", False)
    setattr(opt, "with_noise", False)
    setattr(opt, "is_limit_class", False)
    setattr(opt, "is_validate", True)
    setattr(opt, "val_ratio", 0.2)

    print(opt.batch_size)

    testLoder = miniLoader(opt, True)
    dataset, datasize = testLoder.get_data()

    print(datasize)

    idx = 666
    for i_batch , (inputs, labels, noise_mark) in enumerate(dataset):
        # print(len(dataset), 50000/64)
        # print(i_batch)
        if i_batch == idx:
            print(type(inputs), inputs.shape)
            print(type(labels), labels.shape)

            inputs_np = torch.squeeze(inputs, dim=0)
            tmp = inputs_np.numpy().transpose(1,2,0)
            print(tmp.shape)
            cv2.imshow('Color image', tmp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if i_batch ==idx:
            break
    
    # setattr(opt, "batch_size", 128)
    # print(opt.batch_size)
    
    print("happy hacking")
