import os
import pickle
import numpy as np
from copy import deepcopy


def unique_list_list(list_list):
    r'''
    return np.ndarray of unique elements in the list of lists
    Assuming any two indices lists in index_lists are disjoint
    '''
    List = []
    total_num = 0
    for l in list_list:
        List += l
        total_num += len(l)
    num_unique_array = np.unique(List).shape[0]
    assert num_unique_array == total_num, "contain duplicated elements"
    return num_unique_array


# NOTE use new method in generating noise label; using noise matrix
def add_noise_by_mat(original_labels, noise_matrix, noise_label_index = None, random_seed = 10086):
    r'''
    original_labels --- np array; (n,)
    noise_matrix --- np array; (n_class, n_class)
    noise_label_index --- np array; (m,)
    ---------
    noise_matrix[i,j] := P(y_hat = j | y = i);
    '''
    assert isinstance(original_labels, np.ndarray)
    random = np.random.RandomState(random_seed)
    if noise_label_index is None:
        noise_label_index = np.arange(original_labels.shape[0])

    num_classes = noise_matrix.shape[0]
    noise_label = deepcopy(original_labels)
    for idx in noise_label_index:
        target = original_labels[idx]
        if noise_matrix[target, target] != 1.0:
            noise_label[idx] = random.choice(num_classes, 1, p=noise_matrix[target])
    
    return noise_label

# NOTE below not use the noise matrix explicitly
def build_noise_dicts(noise_rates, index_lists):
    r'''
    prepare noise_labels_dict & noise_rates_dict:
    e.g.
        for index_lists[i] = [l, m, n] & noise_rates[i] = [Pl, Pm, Pn]
        one has
            noise_labels_dict[k] = [l, m, n] for k = l, m, n
            noise_rates_dict[l] = [1-Pm-Pn, Pm, Pn]
            noise_rates_dict[m] = [Pl, 1-Pl-Pn, Pn]
            noise_rates_dict[n] = [Pl, Pm, 1-Pl-Pm]
    '''
    len_noise_rates = len(noise_rates)
    len_index_lists = len(index_lists)
    if len_noise_rates != 1:
       assert len_noise_rates == len_index_lists, "noise_rates & index_lists not match"

    noise_labels_dict = {}
    noise_rates_dict = {}
    for i, indices_list in enumerate(index_lists):
        rates = noise_rates[0] if len_noise_rates == 1 else noise_rates[i]
        noise_rates_sum = np.sum(rates)
        for j, index in enumerate(indices_list):
            noise_labels_dict[index] = indices_list
            noise_rates_dict[index] = [a for a in rates]
            noise_rates_dict[index][j] = 1. - noise_rates_sum + rates[j]

    return noise_labels_dict, noise_rates_dict

def add_noise_sparse_specified(original_labels, noise_rates, index_lists, 
                               noise_labels_index = None, random_seed = 10086):
    r'''
    index_lists --- list of noise label lists
    noise_rates --- list of noise rate lists
    ------
    e.g.
        index_lists[i] = [l, m, n] & noise_rates[i] = [Pl, Pm, Pn]
        this means
            classes l, m, n are flipped with in {l, m, n} &
            Pl is P(y_hat = l | y = k), k = m, n; likewise
            Pm is P(y_hat = m | y = k), k = l, n;
            Pn is P(y_hat = n | y = k), k = m, l;
        &
            noise_labels_dict[k] = [l, m, n] for k = l, m, n
            noise_rates_dict[l] = [1-Pm-Pn, Pm, Pn]
            noise_rates_dict[m] = [Pl, 1-Pl-Pn, Pn]
            noise_rates_dict[n] = [Pl, Pm, 1-Pl-Pm]
    '''
    assert isinstance(original_labels, np.ndarray)
    random = np.random.RandomState(random_seed)
    if noise_labels_index is None:
        noise_labels_index = np.arange(original_labels.shape[0])

    num_classes = np.unique(original_labels).shape[0]
    num_noise_classes = unique_list_list(index_lists)
    assert num_noise_classes <= num_classes

    noise_labels_dict, noise_rates_dict = build_noise_dicts(noise_rates, index_lists)
    noise_labels = deepcopy(original_labels)
    for idx in noise_labels_index:
        label = original_labels[idx]
        if label in noise_labels_dict.keys():
            nld = noise_labels_dict[label]
            nrd = noise_rates_dict[label]
            noise_labels[idx] = random.choice(nld, 1, p = nrd)

    return noise_labels

def add_noise_sparse_random(original_labels, noise_upper_bound, 
                            index_lists = None, noise_labels_index = None, random_seed = 10086):
    r'''
    Randomly generate noise_rates, index_lists and call `add_noise_sparse_specified`
    ------
    noise_upper_bound --- float, upper bound for error ratea
    '''
    assert isinstance(original_labels, np.ndarray)
    random = np.random.RandomState(random_seed + 10086)

    # assuming original_labels contain all the classes
    num_classes = np.unique(original_labels).shape[0]
    if index_lists is None:
        index_lists = [[cl for cl in range(num_classes)]]

    noise_rates = []
    for idx_list in index_lists:
        num_idx = len(idx_list)
        rates = random.uniform(0., noise_upper_bound, size=(num_idx,))
        assert any(1. - np.sum(rates) + rates >= 0.0)
        noise_rates.append([a for a in list(rates)])

    return add_noise_sparse_specified(original_labels, noise_rates, index_lists, 
                                      noise_labels_index, random_seed)

# for cases with large number of classes, e.g. ImageNet
def add_noise_sparse(original_labels, noise_mode = "specified", **kwargs):
    if noise_mode == "specified":
        noise_rates = kwargs["noise_rates"]
        index_lists = kwargs["index_lists"]
        noise_labels_index = kwargs["noise_labels_index"] if "noise_labels_index" in kwargs.keys() else None
        random_seed = kwargs["random_seed"] if "random_seed" in kwargs.keys() else 10086
        noise_labels = add_noise_sparse_specified(original_labels, noise_rates, index_lists, 
                                                  noise_labels_index, random_seed)
    elif noise_mode == "random":
        noise_upper_bound = kwargs["noise_upper_bound"]
        index_lists = kwargs["index_lists"] if "index_lists" in kwargs.keys() else None
        noise_labels_index = kwargs["noise_labels_index"] if "noise_labels_index" in kwargs.keys() else None
        random_seed = kwargs["random_seed"] if "random_seed" in kwargs.keys() else 10086
        noise_labels = add_noise_sparse_random(original_labels, noise_upper_bound, 
                                               index_lists, noise_labels_index, random_seed)
    else:
        raise ValueError("invalid mode for adding noise")
    return noise_labels

def add_noise_random_simple(original_labels, noise_rate, random_seed=10086):
    r'''
    For all the orignal labels, randomly flip the original label into a possible label,
    flipping rate = noise_rate
    '''
    assert isinstance(original_labels, np.ndarray)
    random = np.random.RandomState(random_seed + 10086)
    random2 = np.random.RandomState(random_seed + 2333)

    num_classes = np.unique(original_labels).shape[0]
    noise_labels = deepcopy(original_labels)
    for idx in range(original_labels.shape[0]):
        rdn = random2.uniform()
        if rdn <= noise_rate:
            noise_labels[idx] = random.choice(num_classes, 1)
    return noise_labels

def save_noise_mat(num_classes, noise_rates, index_lists, fpath):
    noise_mat = np.zeros((num_classes, num_classes))
    noise_labels_dict, noise_rates_dict = build_noise_dicts(noise_rates, index_lists)
    for row, col_list in noise_labels_dict.items():
        for idx, col in enumerate(col_list):
            noise_mat[row, col] = noise_rates_dict[row][idx]
    with open(fpath, 'wb') as f:
        pickle.dump(noise_mat, f)
    print(f"noise matrix saved to {fpath}") 


if __name__ == "__main__":
    pass
    #NOTE uncomment old codes before running below tests
    # nm1 = build_matrix_pair(np.array([0.2, 0.6]))

    # llist = [[0, 2],
    #          [3, 5],
    #          [4, 7],
    #          [6, 8],
    #          [1, 9]]
    # nm2 = build_noise_matrix(noise_rates=[[0.2, 0.6]], index_lists=llist, num_classes=10)

    # assert np.abs(np.sum(nm1.transpose() - nm2)) < 1e-10

    # noise_list = np.array([0.02, 0.03, 0.01, 0.02, 0.05, 0.03, 0.06, 0.01, 0.02, 0.05])
    # nm3 = build_matrix(noise_list, 10086, 0.5, uniform = True)

    # nm4 = build_noise_matrix(noise_rates=[[0.02, 0.03, 0.01, 0.02, 0.05, 0.03, 0.06, 0.01, 0.02, 0.05]],
    #                          index_lists=[[0,1,2,3,4,5,6,7,8,9]], num_classes=10)

    # assert np.abs(np.sum(nm3.transpose() - nm4)) < 1e-10

    # num_classes = 10
    # original_labels = np.random.choice(num_classes, size=(100,))
    # assert np.unique(original_labels).shape[0] == num_classes
    # noise_rates = [[0.2, 0.2, 0.2], [0.1, 0.2, 0.3]]
    # # noise_rates = [[0.1, 0.2, 0.3]] # rates are "uniform" for different lists
    # index_lists = [[0, 1, 3], [2, 6, 8]]
    # n1 = add_noise_sparse(original_labels, noise_rates=noise_rates, index_lists=index_lists)

    # nm = build_noise_matrix(noise_rates=noise_rates, index_lists=index_lists, num_classes=num_classes)
    # # print(nm)
    # n2 = add_noise(original_labels, nm)

    # assert np.abs(np.sum(n1 - n2)) < 1e-10, print(n1 - n2)