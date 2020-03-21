import os
import sys
import pandas as pd
import datetime
import numpy as np

from .plotutil import *

# np.random.seed()

class Logger:
    def __init__(self, opt, json_path, outputfile=None):
        self._name = "logger"
        self._log_root = opt.log_root
        self._plot_dir = "plots"
        self._data_dir = "data"
        self._opt = opt

        if outputfile is None:
            outputfile = sys.stdout
        self._set_directories(json_path, outputfile)

    def _set_directories(self, json_path, outputfile):
        '''
        1. if given json file, means either continue training or 
            load trained model for validation. Use it as log_path
        2. if json_path == None, create new log_path

        log_path will be the directory for storing:
            * experiment options
            * trained model
            * results: figures & data
        '''
        # set log root dir
        if json_path is None:
            # for new experiment, create relevant directories
            dir_name_list = [self._log_root,
                             f"{self._opt.dataset}",
                             f"{self._opt.netARCH}",
                             f"{self._opt.exp_name}"]
            temp_str = os.path.join(*dir_name_list)
            self.log_path = f"{temp_str}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"

            #resolve possible conflict
            if os.path.exists(self.log_path):
                old_path = self.log_path
                rand = np.random.RandomState()
                self.log_path = f"{old_path}_{rand.randint(99999):0>5}"

            os.makedirs(self.log_path)

        else:
            # if valid json_path, using existed directories
            self.log_path = json_path
            if not os.path.exists(self.log_path):
                raise ValueError(f"required path,{json_path}, not exist")

        # set plot dir
        self._plot_path = os.path.join(self.log_path, self._plot_dir)

        # set data dir
        self._data_path = os.path.join(self.log_path, self._data_dir)
        if not os.path.exists(self._data_path):
            os.makedirs(self._data_path)

        print("\n{}".format(12*"---"), flush=True, file=outputfile)
        print(f"Working at log_path: {self.log_path}", flush=True, file=outputfile)

    def log(self, i_epoch, fname, **kwargs):
        '''
        record training accuracy and loss & validation accuracy and loss
        '''
        DictKeyMap = {
            "alpha":        "Present Alpha",
            "train_acc":    "Training ACC",
            "train_loss":   "Training Loss",
            "n_val_acc":    "Noise Validation ACC",
            "n_val_loss":   "Noise Validation Loss",
            "val_acc":      "Clean Validation ACC",
            "val_loss":     "Clean Validation Loss",
            "t_conf_mat":   "Training Confusion Matrix",
            "t_conf_mat_c": "Clean Training Confusion Matrix",
            "t_conf_mat_n": "Noise Training Confusion Matrix",
            "nv_conf_mat":  "Noisy Validation Confusion Matrix",
            "v_conf_mat":   "Clean Validation Confusion Matrix"
        }

        fpath = os.path.join(self._data_path, fname)

        Dict ={"Epoch": [i_epoch]}
        for var in kwargs.keys():
            Dict[DictKeyMap[var]] = [kwargs[var]]
        
        df = pd.DataFrame.from_dict(Dict)

        if os.path.isfile(fpath):
            df.to_csv(fpath, index=False, mode='a', header=False)
        else:
            df.to_csv(fpath, index=False, mode='w')

    def plot(self, max_epoch, fname, ic=None, k=None):
        fpath = os.path.join(self._data_path, fname)
        df = pd.read_csv(fpath)
        adict = df.to_dict(orient='list')

        # plot record of accuracy
        acc_fname = "accuracy.jpg" if (ic is None) and (k is None) else f"accuracy_cf{ic}_step{k}.jpg"
        train_acc = adict["Training ACC"]
        val_acc = adict["Noise Validation ACC"]
        test_acc = adict["Clean Validation ACC"]
        str_dict = {
            "fpath": os.path.join(self._data_path, acc_fname),
            "ylabel": "Accuracy",
            "legend1": "training accuracy",
            "legend2": "noise validate accuracy",
            "legend3": "clean validate accuracy"
        }
        plot_training_records(str_dict, xmax=max_epoch, y1=train_acc, y2=val_acc, y3=test_acc)
        
        # plot record of loss
        # loss_fname = "loss.jpg" if (ic is None) and (k is None) else f"loss_cf{ic}_step{k}.jpg"
        # train_loss = adict["Training Loss"]
        # val_loss = adict["Noise Validation Loss"]
        # test_loss = adict["Clean Validation Loss"]
        # lgd1 = "training loss (peer)" if self._opt.is_peerloss else "training loss"
        # str_dict = {
        #     "fpath": os.path.join(self._data_path, loss_fname),
        #     "ylabel": "Loss",
        #     "legend1": lgd1,
        #     "legend2": "noise validate loss",
        #     "legend3": "clean validate loss"
        # }
        # plot_training_records(str_dict, xmax=max_epoch, y1=train_loss, y2=val_loss, y3=test_loss)



if __name__ == "__main__":
    pass