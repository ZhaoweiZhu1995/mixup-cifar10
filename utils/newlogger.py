import os
import sys
import json
import pandas as pd
import datetime
import numpy as np


class Probe(object):
    r"""
    define a moethod to record (write/read) values by specifying tag
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.datafile_path = os.path.join(self.data_path, "data.json")#?

        # initialize dictionary for holding data temparaily 
        self.data_pool = {}

        self.step = -1

    def update_step(self):
        self.step += 1

    def add_data(self, tag, data, global_step=None, walltime=None):

        step = self.step if global_step is None else global_step
        if step not in self.data_pool.keys():
            self.data_pool[step] = {}

        if tag not in self.data_pool[step].keys():
            self.data_pool[step][tag] = []

        self.data_pool[step][tag] += data if isinstance(data, list) else [data]

    def cleanup_data_pool(self):
        del self.data_pool
        self.data_pool = {}

    def flush(self):
        r'''flush to file & cleanup'''
        pass

class Logger(object):
    def __init__(self, opt, json_path, outputfile=None):
        self._name = "logger"
        self.log_root = opt.log_root
        self._plot_dir = "plots"
        self._data_dir = "data"
        self.opt = opt

        if outputfile is None:
            outputfile = sys.stdout
        self._dump_info(outputfile)
        self._set_directories(json_path, outputfile)

        #initialize data probe for recording data
        self._outfile = outputfile
        self.probe = Probe(self._data_path)
        self.cfg = {}

    @property
    def logdir(self):
        return self.log_path
    
    def _dump_info(self, outputfile):
        print(f"\n{'Dataset:':>10} {self.opt.dataset}"\
              f"\n{'NetATCH:':>10} {self.opt.netARCH}"\
              f"\n{'Optimizer:':>10} {self.opt.optimizer}"\
              f"\n{'Lossfunc:':>10} {self.opt.lossfunc}",\
              flush=True, file=outputfile)

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
            dir_name_list = [self.log_root,
                             self.opt.dataset,
                             self.opt.netARCH,
                             self.opt.exp_name]
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
    
    def add_cfg(self, tag, val):
        self.cfg[tag] = val
    
    def add_data(self, tag, data, global_step=None, walltime=None):
        r'''interface for recording data'''
        self.probe.add_data(tag, data, global_step, walltime)

    def display_acc_loss_batch(self, i_batch, batch_size, log_type="train"):
        if i_batch == 0:
            self.p = 0.1

        total_num_batch = self.cfg[f"{log_type}/batch_num"]
        if i_batch >= int(self.p * total_num_batch):

            loss = self.probe.data_pool[self.probe.step][f"{log_type}/loss"][-1]
            corrects = self.probe.data_pool[self.probe.step][f"{log_type}/corrects"][-1]

            print(f"[{i_batch:3}/{total_num_batch:3}], current batch loss: {loss:.5f}",\
                f"with corrects {corrects:3.0f}/{batch_size} ({corrects/float(batch_size):.5f})",\
                flush=True, file=self._outfile)
            
            self.p += 0.1

    def log_acc_loss(self, acc_tag, loss_tag, is_save=True):
        log_type = acc_tag.split("/")[0]
        assert log_type == loss_tag.split("/")[0], "not matching types"

        corrects = self.probe.data_pool[self.probe.step][acc_tag]
        loss = self.probe.data_pool[self.probe.step][loss_tag]

        avg_acc = np.sum(corrects) / float(self.cfg[f"{log_type}/datasize"])
        avg_loss = np.sum(loss) / float(self.cfg[f"{log_type}/batch_num"])

        print(f"\nFor epoch {self.probe.step}, average {log_type} "\
              f"loss = {avg_loss:.5f}; accuracy = {avg_acc:.5f};",\
              flush=True, file=self._outfile)

        self.avg_accuracy = avg_acc #interface to model, update after computing

    def log_confmat(self, tag, print_lim = 10, is_save=True):
        log_type = tag.split("/")[0]
        idx_pairs = self.probe.data_pool[self.probe.step][tag]
        conf_mat = np.zeros((self.opt.num_classes, self.opt.num_classes), dtype = int)
        for l, p in idx_pairs:
            conf_mat[l, p] += 1

        if print_lim > self.opt.num_classes:
            print_lim = min(self.opt.num_classes, 10)
        print(f'total {log_type} conf_mat is: \n{conf_mat[:print_lim, :print_lim]}', \
                flush=True, file=self._outfile)
        print(f'check sum: {np.sum(conf_mat)}', flush=True, file=self._outfile)

    #     if is_save:
    #         self.save_csv(self.probe.step, conf_mat)

    # def save_csv(self, i_epoch, fname, **kwargs):
    #     '''
    #     record training accuracy and loss & validation accuracy and loss
    #     '''
    #     DictKeyMap = {
    #         "alpha":        "Present Alpha",
    #         "train_acc":    "Training ACC",
    #         "train_loss":   "Training Loss",
    #         "n_val_acc":    "Noise Validation ACC",
    #         "n_val_loss":   "Noise Validation Loss",
    #         "val_acc":      "Clean Validation ACC",
    #         "val_loss":     "Clean Validation Loss",
    #         "t_conf_mat":   "Training Confusion Matrix",
    #         "t_conf_mat_c": "Clean Training Confusion Matrix",
    #         "t_conf_mat_n": "Noise Training Confusion Matrix",
    #         "nv_conf_mat":  "Noisy Validation Confusion Matrix",
    #         "v_conf_mat":   "Clean Validation Confusion Matrix"
    #     }

    #     fpath = os.path.join(self._data_path, fname)

    #     Dict ={"Epoch": [i_epoch]}
    #     for var in kwargs.keys():
    #         Dict[DictKeyMap[var]] = [kwargs[var]]
        
    #     df = pd.DataFrame.from_dict(Dict)

    #     if os.path.isfile(fpath):
    #         df.to_csv(fpath, index=False, mode='a', header=False)
    #     else:
    #         df.to_csv(fpath, index=False, mode='w')

    # def _save_base_peer(self, base, peer, save_path):
    #     Dict = {}
    #     Dict["base loss value"] = [base]
    #     Dict["peer term value"] = [peer]
    #     df = pd.DataFrame.from_dict(Dict)
    #     if os.path.isfile(save_path):
    #         df.to_csv(save_path, index=False, mode='a', header=False)
    #     else:
    #         df.to_csv(save_path, index=False, mode='w')