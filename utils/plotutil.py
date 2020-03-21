import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
import numpy as np
import torch
import torch.nn as nn
import os
from lossfuncs.bitemploss import tempered_sigmoid

'''
plot:
--- training loss, validate loss, test loss
--- training accuracy, validate accuracy, test accuracy
'''
def plot_training_records(str_dict, **kwargs):
    '''
    expect xmax, y1, y2, y3
    '''
    xmax = kwargs["xmax"]
    x = np.arange(xmax)
    y1 = np.array(kwargs["y1"])
    label1 = str_dict["legend1"]
    y2 = np.array(kwargs["y2"])
    label2 = str_dict["legend2"]
    y3 = np.array(kwargs["y3"])
    label3 = str_dict["legend3"]
    fpath = str_dict["fpath"]
    ylabel = str_dict["ylabel"]

    fig = plt.figure(figsize=(7,5))
    ax  = fig.add_subplot(1,1,1)

    ax.plot(x, y1, label=label1)
    ax.plot(x, y2, label=label2)
    ax.plot(x, y3, label=label3)

    ax.set_xlabel('number of epoch', fontsize = 19)
    ax.set_ylabel(ylabel, fontsize = 19)
    ax.legend()

    fig.savefig(fpath, format='jpeg')
    plt.close(fig)

def plot_decision_boundary(train_data, train_label, model, t=None, figtitle=None, figname=None):
    X_mesh, Y_mesh = np.mgrid[0:256:150j, 0:256:150j]
    X_flat = np.array([X_mesh.ravel()])
    Y_flat = np.array([Y_mesh.ravel()])
    XY_eval = np.concatenate((X_flat, Y_flat), axis = 0).T

    sigmoid = nn.Sigmoid()
    feature = torch.tensor(XY_eval, dtype=torch.float)
    with torch.set_grad_enabled(False):
        outputs, _ = model.predict(feature)
        if t is None:
            probs = sigmoid(outputs)
        else:
            probs = tempered_sigmoid(outputs, t, 5)

    Z = probs.numpy()
    Z_mesh = Z.reshape(X_mesh.shape)

    colors = ['#6495ED','#64b4ed','#64dded',
              '#dca314','#dc6b14','#dc4d14']
    cmap_test = ListedColormap(colors)
    levels_test = [0.1,0.3,0.5,0.7,0.9]
    norm = BoundaryNorm(levels_test, ncolors=cmap_test.N, clip=True)

    plt.figure(figsize=(8, 8))
    plt.pcolormesh(X_mesh, Y_mesh, Z_mesh, cmap=cmap_test, norm=norm)
    c = plt.contour(X_mesh, Y_mesh, Z_mesh, colors=['k', 'k', 'k'],
                    linestyles=['--', '-', '-.'], levels=[0.3, 0.5, 0.7])
    # c = plt.contour(X_mesh, Y_mesh, Z_mesh, colors=['k'], linestyles=['--'], levels=[0.5])
    try:
        c.collections[0].set_label("Pred Prob = 0.3")
        c.collections[1].set_label("Decision Boundary (0.5)")
        c.collections[2].set_label("Pred Prob = 0.7")
    except:
        c.collections[0].set_label("tmp boundary")

    data_c0 = train_data[(train_label==0)].numpy()
    data_c1 = train_data[(train_label==1)].numpy()
    
    # plt.scatter(data_c1[:,0], data_c1[:,1], facecolors='#B22222', edgecolors='k', label = "Class 1")
    # plt.scatter(data_c0[:,0], data_c0[:,1], facecolors='blue', edgecolors='k', label = "Class 0")
    plt.scatter(data_c1[:,0], data_c1[:,1], facecolors='#B22222', s = 12, linewidths = 0.5, edgecolors='k')
    plt.scatter(data_c0[:,0], data_c0[:,1], facecolors='blue', s = 12, linewidths = 0.5, edgecolors='k')
    # plt.scatter(data_c1[:,0], data_c1[:,1], color='#B22222', marker = '.')
    # plt.scatter(data_c0[:,0], data_c0[:,1], color='blue', marker = '.')
    # plt.plot([-10,256+10],[128,128], linestyle='--', c = '0.75')
    plt.xlim([0,256])
    plt.ylim([0,256])

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize = 14, loc = 'lower right')

    if figtitle is None:
        figtitle = "Decision Boundary"
    plt.title(figtitle, fontsize=16, y=1.03)
    if figname is None:
        figname = "testfig.png"
    plt.savefig(os.path.join("./figures", figname), dpi=200)
    plt.close()



if __name__ == "__main__":
    import sys
    from os import path
    sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

    import models
    from datamanage.dataloader import miniLoader
    from utils.functions import get_parameters

    param_dict = {"--is_train": True,
                    "--is_plot_results": True,
                    "--exp_name": "reproduce_manfred_demo",
                    "--dataset": "ManfredDemo",
                    "--ARCH": "bitemp_dense",
                    "--num_classes": 2,
                    '--lossfunc': "bitempered_binary",
                    "--T1": 0.8,
                    "--T2": 2.1,
                    "--batch_size": 256,
                    "--is_normalize": True,
                    "--gpu_idx": '0'
                    }
    opt = get_parameters(param_dict, None)

    loader = miniLoader(opt, False)
    loader.get_data()
    data, label, _ = loader.get_raw_data()

    log_path = "logs/ManfredDemo/bitemp_dense/BiTB__t1-0.6_t2-2.2_2020-01-17-17-08"
    model = models.BasePeerModel(log_path, opt)
    model.load(is_predict=True)

    plot_decision_boundary(data, label, model)