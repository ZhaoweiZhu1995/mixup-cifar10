import os
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.dirname( path.abspath(__file__) ) ) ) )

from PIL import Image

import torch
from torch.utils.data import Dataset

import numpy as np
import random
import pickle


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)

def load_label(label_file_path):
    '''
    I adopte .pt rather .pth according to this discussion:
    https://github.com/pytorch/pytorch/issues/14864
    '''
    #NOTE presently only use for load manual training label
    noise_label = torch.load(label_file_path)
    if isinstance(noise_label, dict):
        return noise_label['noise_label_train']
    else:
        return noise_label

def get_class_size(images, original_label, num_classes):
    idx_each_class = [[] for i in range(num_classes)]
    for i in range(len(images)):
        # idx_each_class[images[i][1]].append(i)
        idx_each_class[original_label[i]].append(i)
    for i in range(num_classes):
        random.shuffle(idx_each_class[i])
    class_size = [len(idx_each_class[i]) for i in range(num_classes)]
    print(f'The original data size in each class is {class_size}')
    return class_size, idx_each_class

def set_class_dist(images, original_label, num_classes, classes_dist):
    selected_idx = []
    class_size, idx_each_class = get_class_size(images, original_label, num_classes)
    ub_list = np.array(classes_dist[:-1])
    ub_list = classes_dist[-1] * (ub_list/np.sum(ub_list))
    ub_list = ub_list.astype(int)
    for i in range(len(class_size)):
        if ub_list[i]>class_size[i]:
            raise ValueError(f'Too much training data in a class {i}')
        else:
            selected_idx += list(np.array(idx_each_class[i])[:ub_list[i]])
            random.shuffle(selected_idx)
            print(f'accumulated length is {len(selected_idx)}')
    truncated_images = []
    truncated_labels = []
    for idx in selected_idx:
        truncated_images.append(images[idx])
        truncated_labels.append(original_label[idx])
    return truncated_images, truncated_labels

def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None,
                 label_file_path=None, chosen_classes=None, selected_idx=None, classes_dist=None):
    r'''
    NOTE the preprocessing is included here
    '''
    if selected_idx is not None:
        selected_idx_set = set(selected_idx)

    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    
    # if use a stand alone label file; say the noise labels
    #NOTE Please make sure the oreder of data is the same in the label file
    if label_file_path is not None:
        flabel = load_label(label_file_path)

    # prepare the target-label mapping
    if chosen_classes is not None:
        #TODO
        if label_file_path is not None:
            print("May cause error when noise labels have members not in the chosen_classes")
            raise NotImplementedError

        label_map = {}
        idx = -1
        for target in sorted(class_to_idx.keys()):
            if class_to_idx[target] in chosen_classes:
                idx += 1
                label_map[target] = idx
    else:
        label_map = class_to_idx

    num_classes = len(label_map.keys())

    original_label = []
    fidx = -1
    # sub_class_idx = {i: [] for i in range(14)}
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                fidx += 1 # update the current index for the data file
                if is_valid_file(path) and \
                   ( (selected_idx is None) or (fidx in selected_idx_set) ):
                    if target in sorted(label_map.keys()):
                        original_label.append(label_map[target])
                        label = label_map[target] if label_file_path is None else flabel[fidx]
                        item = (path, label)
                        images.append(item)
                        # sub_class_idx[ label_map[target] ].append(fidx)

    # chosen_idx = []
    # for idx in sub_class_idx.keys():
    #     chosen_class_idx = np.random.choice(sub_class_idx[idx], size=18976, replace=False).tolist()
    #     chosen_idx += chosen_class_idx
    # with open("./C1M_selected_idx_balance.pkl", "wb") as f:
    #     pickle.dump(chosen_idx, f)
    # with open("./C1M_class_idx_dist.pkl", "wb") as f:
    #     pickle.dump(sub_class_idx, f)

    # set distribution over diff classes; 
    if classes_dist is not None:
        images, original_label = set_class_dist(images, original_label, num_classes, classes_dist)
    else:
        get_class_size(images, original_label, num_classes)

    return images, original_label

class DatasetFolder(Dataset):
    def __init__(self, root, loader, extensions=None, transform=None,
                 is_valid_file=None, label_file_path=None,
                 selected_idx=None, chosen_classes=None, classes_dist=None, is_mixup=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.label_file_path = label_file_path
        self.selected_idx = selected_idx
        self.chosen_classes = chosen_classes
        self.classes_dist = classes_dist
        self.is_mixup = is_mixup

        classes, class_to_idx = self._find_classes(self.root)
        samples, original_labels = make_dataset(self.root, class_to_idx, extensions, is_valid_file, 
                               label_file_path, chosen_classes, selected_idx, classes_dist)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.original_labels = torch.tensor(original_labels)
        self.activ_class = torch.unique(self.original_labels).shape[0]
        if (chosen_classes is not None) and (self.activ_class < len(chosen_classes)):

            print(f"\n!!!\nonly {self.activ_class}/{len(chosen_classes)} classes present in selected samples\n!!!\n")

        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        # self.targets = [s[1] for s in samples] #not used presently
    
    @property
    def num_classes(self):
        return self.activ_class

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.is_mixup:
            # assume self.transform is not None
            if isinstance(index, list) or isinstance(index, np.ndarray) or isinstance(index, torch.Tensor):
                sample_list_1 = []
                sample_list_2 = []
                target_list = []
                for idx in index:
                    path, target_s = self.samples[idx]
                    sample_s = self.loader(path)
                    sample_s_1 = self.transform(sample_s)
                    sample_s_2 = self.transform(sample_s)
                    # assume sample_s are torch.Tensor
                    sample_list_1.append(sample_s_1)
                    sample_list_2.append(sample_s_2)
                    # usually target_s are int
                    target_list.append(target_s)
                sample_1 = torch.stack(sample_list_1, dim=0)
                sample_2 = torch.stack(sample_list_2, dim=0)
                target = torch.tensor(target_list).long()
            else:
                path, target = self.samples[index]
                sample = self.loader(path)
                sample_1 = self.transform(sample)
                sample_2 = self.transform(sample)
                target = torch.tensor(target).long()

            return sample_1, sample_2, target, self.original_labels[index], index
        else:
            if isinstance(index, list) or isinstance(index, np.ndarray) or isinstance(index, torch.Tensor):
                sample_list = []
                target_list = []
                for idx in index:
                    path, target_s = self.samples[idx]
                    sample_s = self.loader(path)
                    if self.transform is not None:
                        sample_s = self.transform(sample_s)
                    # assume sample_s are torch.Tensor
                    sample_list.append(sample_s)
                    # usually target_s are int
                    target_list.append(target_s)
                sample = torch.stack(sample_list, dim=0)
                target = torch.tensor(target_list).long()
            else:
                path, target = self.samples[index]
                sample = self.loader(path)
                if self.transform is not None:
                    sample = self.transform(sample)
                target = torch.tensor(target).long()

            return sample, target, self.original_labels[index], index

    def __len__(self):
        return len(self.samples)



IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None,
                 loader=default_loader, is_valid_file=None, label_file_path=None,
                 selected_idx=None, chosen_classes=None, classes_dist=None, is_mixup=False):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          is_valid_file=is_valid_file,
                                          label_file_path=label_file_path,
                                          selected_idx=selected_idx, 
                                          chosen_classes=chosen_classes, 
                                          classes_dist=classes_dist,
                                          is_mixup=is_mixup)
        # self.imgs = self.samples # not used presently



if __name__ == "__main__":
    dir = "datasets/"
    print(os.path.expanduser(dir))
    fn = "asdf.pdd"
    ext = ("jpg", "pdd")
    print(has_file_allowed_extension(fn, ext))