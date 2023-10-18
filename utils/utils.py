from sklearn.metrics import f1_score, matthews_corrcoef
from scipy import stats
import numpy as np
class metric:
    def flat_accuracy(preds, labels):
        return np.sum(preds == labels) / len(labels)
    
    #or average = "weighted"
    def F1_score(preds, labels, average = 'binary'):
        try:
            return f1_score(labels, preds, average=average)
        except Exception as e:
            print(e)
            return 0

    def Corr(preds, labels, num = 2):
        try:
            return matthews_corrcoef(labels, preds)
        except Exception as e:
            print(e)
            return 0
    
    def Spear(preds, labels):
        try:
            return stats.spearmanr(preds, labels).correlation
        except Exception as e:
            print(e)
            return 0
    
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
class process_dataset:
    def BERTDataLoader(data, batch = 16, sampler = "random"):
        data = TensorDataset(data['input_ids'], 
                             data['attention_mask'], 
                             data['token_type_ids'], 
                             data['labels'])
        if sampler == "random":
            data_sampler = RandomSampler(data)
        else:
            data_sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler = data_sampler, batch_size = batch)
        return dataloader

    def DataLoader(data, batch = 16, sampler = "random"):
        data = TensorDataset(data['input_ids'],
                             data['labels'])
        if sampler == "random":
            data_sampler = RandomSampler(data)
        else:
            data_sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler = data_sampler, batch_size = batch)
        return dataloader

import torch
import os
"""
multi GPU then
if device.type == 'cuda' and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
"""
def get_device(use_gpu = True, CUDA_VISIBLE_DEVICES = "0, 1, 2"):
    if use_gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device
    
    device = torch.device("cpu")
    return device
   
import datetime
import time

def format_time(elapsed):
    # 반올림
    elapsed_rounded = int(round((elapsed)))
    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))

class record_training:
    def __init__(self, epochs = 0, data_len = 0, OUTPUT_DIR = "./", file_name = "base"):
        self.before_parameter = {}
        self.total_change = {}
        self.output_dir = OUTPUT_DIR
        self.file_name = file_name
        self.total_loss = 0
        self.t0 = 0
        
        self.logits = np.array([])
        self.labels = np.array([])
        

        self.epochs = epochs
        self.data_len = data_len
        
        with open(OUTPUT_DIR + "parameter_" + file_name + ".txt", 'w') as f:
            f.write('Record parameter changes\n')
        with open(OUTPUT_DIR + "result_" + file_name + ".txt", 'w') as f:
            f.write('Record result\n')

    def save_before_paramter(self, model):
        for name, param in model.named_parameters():
            self.before_parameter[name] = param.clone().detach().requires_grad_(False)
            self.total_change[name] = 0
            
    def save_parameter_change(self, model):
        for name, param in model.named_parameters():
            diff = param.clone().detach().requires_grad_(False) - self.before_parameter[name]
            self.total_change[name] += diff
            
    def record_parameter_change(self, model, epoch_i):
        with open(self.output_dir + "parameter_" + self.file_name + ".txt", 'a') as f:
            f.write('\nEpochs : %d\n'%epoch_i)
        for name, param in model.named_parameters():
            with open(self.output_dir + "parameter_" + self.file_name + ".txt", 'a') as f:
                f.write(name + ' : {}\n'.format(torch.sum(self.total_change[name])))
    
    #새로운 epoch 진입
    def init_epoch(self, epoch_i = 0):
        self.total_loss = 0
        self.t0 = time.time()
        with open(self.output_dir + "result_" + self.file_name + ".txt", 'a') as f:
            f.write("\n" + '======== Epoch {:} / {:} ========\n'.format(epoch_i + 1, self.epochs) +'Training...\n')
    
    def save_loss(self, loss):
        self.total_loss += loss
    
    def record_step_loss(self, step):
        elapsed = format_time(time.time() - self.t0)
        temp_total = self.total_loss / step
        with open(self.output_dir + "result_" + self.file_name + ".txt", 'a') as f:
            f.write('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.\n'.format(step, self.data_len, elapsed))
            f.write('Loss / step : {0:.2f}\n'.format(float(temp_total)))
            
    def record_epoch_loss(self):
        avg_train_loss = self.total_loss / self.data_len
        with open(self.output_dir + "result_" + self.file_name + ".txt", 'a') as f:
            f.write("\n" + "  Average training loss: {0:.2f}\n".format(float(avg_train_loss)) +"  Training epcoh took: {:}\n".format(format_time(time.time() - self.t0)))
    
    def get_score(self, logits, labels, num_labels):
        if num_labels == 1:
            self.logits  = np.concatenate((self.logits, logits.flatten()), axis=None)
        else:
            if len(logits.shape) != 1:
                self.logits  = np.concatenate((self.logits, np.argmax(logits, axis=-1).flatten()), axis=None)
            else:
                self.logits  = np.concatenate((self.logits, logits), axis=None)
        self.labels  = np.concatenate((self.labels, labels), axis=None)
        self.num_labels = num_labels
    
    def record_metric(self):
        accuracy = metric.flat_accuracy(self.logits, self.labels)
        with open(self.output_dir + "result_" + self.file_name + ".txt", 'a') as f:
            f.write("  Accuracy: {0:.3f}\n".format(accuracy))
            f.write("  F1 score: {0:.3f}\n".format(metric.F1_score(self.logits, self.labels)))
            f.write("  F1 score(macro): {0:.3f}\n".format(metric.F1_score(self.logits, self.labels, average = "macro")))
            f.write("  F1 score(weighted): {0:.3f}\n".format(metric.F1_score(self.logits, self.labels, average = "weighted")))
            f.write("  Mathew's Corr : {0:.3f}\n".format(metric.Corr(self.logits, self.labels)))
            f.write("  Spear Corr : {0:.3f}\n".format(metric.Spear(self.logits, self.labels)))
            f.write("  Validation took: {:}\n".format(format_time(time.time() - self.t0)))
        return accuracy
            
    def init_validation(self):
        with open(self.output_dir + "result_" + self.file_name + ".txt", 'a') as f:
            f.write("\nRunning Validation...\n")
        self.t0 = time.time()
        self.logits = np.array([])
        self.labels = np.array([])
     
    def init_test(self):
        with open(self.output_dir + "result_" + self.file_name + ".txt", 'a') as f:
            f.write("\nRunning Test...\n")
        self.t0 = time.time()
        self.logits = np.array([])
        self.labels = np.array([])
     
    
class gradient_stop:
    def stop(model, layer_number):
        for name, param in model.named_parameters():
            if str(layer_number) in name:
                break
            param.requires_grad = False
        return model
    
    def stop_name(model, layer_name):
        for name, param in model.named_parameters():
            if layer_name in name:
                param.requires_grad = False
            
        return model
    
    def stop_except(model, layer_name):
        for name, param in model.named_parameters():
            if layer_name not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
            
        return model
    
    def stop_original(model, n_cluster = 2, classifier = True):
        for name, param in model.named_parameters():
            flag = True
            for i in range(n_cluster):
                if "_.%d"%i in name:
                    flag = False
                if "classifier" in name and classifier:
                    flag = False
                if "pooler" in name and classifier:
                    flag = False

                    
            if flag:
                param.requires_grad = False
            else:
                param.requires_grad = True
                
            
        return model