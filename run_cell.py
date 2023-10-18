from transformers import BertConfig, AdamW, get_scheduler
import torch
import torch.nn as nn
from Model import CELLBERT as Model
import random
import numpy as np
import argparse

from utils import process_dataset, get_device, record_training, gradient_stop

parser = argparse.ArgumentParser(description='train')

parser.add_argument('--input_dir', type = str, default = None, required=True)
parser.add_argument('--validation_dir', type = str, default = None)
parser.add_argument('--validation_mis_dir', type = str, default = None)#for mnli
parser.add_argument('--output_dir', type = str, default = './')

parser.add_argument('--save_dir', type = str, default = './')
#3 means save model every 3 epochs
parser.add_argument('--save', '-s', type = int, default = -1)
parser.add_argument('--epochs', '-e', type = int, default = 100)
parser.add_argument('--batch_size', '-b', type = int, default = 16)
parser.add_argument('--lr_rate', '-lr', type = float, default = 2e-5)
parser.add_argument('--use_gpu', '-g', default = False, action = 'store_true')
parser.add_argument('--task_name', type = str, required = True)

#fine-tuning model path
parser.add_argument('--final_model', '-fa', default = None)
#additional model num
parser.add_argument('--model_num', '-mn', type = int, default = 2)
parser.add_argument('--cluster', '-cl', type = str, default = "KMeans")


args = parser.parse_args()
INPUT_DIR = args.input_dir
VALID_DIR = args.validation_dir
VALID_MIS_DIR = args.validation_mis_dir
OUTPUT_DIR = args.output_dir
SAVE = args.save
SAVE_DIR = args.save_dir
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LR_RATE = args.lr_rate
GPU = args.use_gpu
Final_Path = args.final_model
TASK_NAME = args.task_name
MODEL_NUM = args.model_num
CLUSTER = args.cluster


def run():
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    global BATCH_SIZE
    DATA_NAME = TASK_NAME  + "_M:" + str(MODEL_NUM) + "_b:" + str(BATCH_SIZE) + "_lr:" + str(LR_RATE)
    if "mnli" in TASK_NAME:
        DATA_NAME = TASK_NAME + "_match" + "_M:" + str(MODEL_NUM) + "_b:" + str(BATCH_SIZE) + "_lr:" + str(LR_RATE)
        DATA_NAME_mis = TASK_NAME + "_mismatch" + "_M:" + str(MODEL_NUM) + "_b:" + str(BATCH_SIZE) + "_lr:" + str(LR_RATE)
        
    train_input = torch.load(INPUT_DIR)
    valid_input = torch.load(VALID_DIR)
    if VALID_MIS_DIR != None:
        valid_mis_input = torch.load(VALID_MIS_DIR)
    
    origin_batch = BATCH_SIZE
    if BATCH_SIZE > 8:
        BATCH_SIZE = 8
    
    train_dataloader = process_dataset.BERTDataLoader(train_input, BATCH_SIZE)
    valid_dataloader = process_dataset.BERTDataLoader(valid_input, BATCH_SIZE)
    if VALID_MIS_DIR != None:
        valid_mis_dataloader = process_dataset.BERTDataLoader(valid_mis_input, BATCH_SIZE)
    
    
    #config from hugging face
    config = BertConfig().from_pretrained("bert-base-uncased")
    config.num_labels = len(set(train_input['labels'].T[0].tolist()))
    config.cluster_name = CLUSTER
    state_dict = torch.load(Final_Path)
        
    model = Model(config)
    model.load_state_dict(state_dict, strict = False)    
    
    device = get_device()
    model.to(device)
    
    #Forwarding for training cluster
    for step, batch in enumerate(train_dataloader):
        #max data size for training cluster is 10K
        if step * BATCH_SIZE > 10000:
            break
        batch = tuple(t.type(torch.LongTensor).to(device) for t in batch)
        b_input_ids, b_input_attention, b_input_type, b_input_label = batch
        with torch.no_grad():
            outputs = model(input_ids = b_input_ids, 
                            attention_mask = b_input_attention,
                            labels = b_input_label,
                            token_type_ids = b_input_type,
                            save_hidden = True)
            
    ####################################################################
    ##  Make additional model and learn cluster for ensemble learning ##
    ####################################################################
    model.make_sub_model(MODEL_NUM, model)
    
    #Freezing base model
    gradient_stop.stop_original(model, MODEL_NUM)
    model.to(device)
    
    optimizer = AdamW(model.parameters(),
                  lr = LR_RATE,
                  eps = 1e-8
                )
    epochs = EPOCHS
    total_steps = len(train_dataloader) * epochs
    scheduler = get_scheduler(name = 'linear',
                              optimizer = optimizer, 
                              num_warmup_steps = 0,
                              num_training_steps = total_steps)
    
    #ex) loss, accuracy, F1 score, training time
    record = record_training(epochs = epochs, data_len = len(train_dataloader), OUTPUT_DIR = OUTPUT_DIR, file_name = DATA_NAME)
    if "mnli" in TASK_NAME:
        record2 = record_training(epochs = epochs, data_len = len(train_dataloader), OUTPUT_DIR = OUTPUT_DIR, file_name = DATA_NAME_mis)
        
    accumulation_step = int(origin_batch / BATCH_SIZE)
    for epoch_i in range(0, epochs):
        record.init_epoch(epoch_i)
        if "mnli" in TASK_NAME:
            record2.init_epoch(epoch_i)
            
        model.zero_grad()
        model.train()
        
        #Train start
        num_batches = 0
        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_input_attention, b_input_type, b_input_label = batch
            b_input_ids = b_input_ids.type(torch.LongTensor).to(device)
            b_input_attention = b_input_attention.type(torch.LongTensor).to(device)
            b_input_type = b_input_type.type(torch.LongTensor).to(device)
            b_input_label = b_input_label.type(torch.LongTensor).to(device)
                
                
            outputs = model(input_ids = b_input_ids, 
                            attention_mask = b_input_attention, 
                            token_type_ids = b_input_type,
                            labels = b_input_label,
                            active_cluster = True,
                            device = device)
            
            loss = outputs[0].mean() / accumulation_step
            loss.backward()
            num_batches += 1
            
            #accumulation
            if num_batches % accumulation_step == 0 or accumulation_step == 0:
                record.save_loss(float(loss))
                if "mnli" in TASK_NAME:
                    record2.save_loss(float(loss))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            
            del b_input_ids
            del b_input_attention
            del b_input_type
            del b_input_label
            del outputs
            torch.cuda.empty_cache()
            
            
            
        record.record_epoch_loss()
        if "mnli" in TASK_NAME:
            record2.record_epoch_loss()
        if SAVE != -1 and epoch_i % SAVE == 0:
            torch.save(model.state_dict(), SAVE_DIR + '/model_%d_'%(epoch_i) + DATA_NAME)
        
        
        #Validation start
        model.eval()
        ##############################################
        ## [1, 0] means : only base model           ##
        ## [0, 1] means : only in-cluster model     ##
        ## [1, 0] means : only out-of-cluster model ##
        ##############################################
        weights = [[1, 0], [0, 1], [0, 0]]
        model_accuracy = []
        for weight in weights:
            ##############################################
            record.init_validation()
            for batch in valid_dataloader:
                b_input_ids, b_input_attention, b_input_type, b_input_label = batch
                b_input_ids = b_input_ids.type(torch.LongTensor).to(device)
                b_input_attention = b_input_attention.type(torch.LongTensor).to(device)
                b_input_type = b_input_type.type(torch.LongTensor).to(device)
                if "stsb" in INPUT_DIR:
                    b_input_label = b_input_label.type(torch.FloatTensor).to(device)
                else:
                    b_input_label = b_input_label.type(torch.LongTensor).to(device)
                
                with torch.no_grad():
                    outputs = model(input_ids = b_input_ids, 
                                    attention_mask = b_input_attention,         
                                    token_type_ids = b_input_type,
                                    active_cluster = True,
                                    mode = "validation",
                                    weighted = weight)
                
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                label_ids = b_input_label.to('cpu').numpy()
            
                record.get_score(logits, label_ids, config.num_labels)
                    
                del b_input_ids
                del b_input_attention
                del b_input_type
                del b_input_label
                del outputs
                torch.cuda.empty_cache()
            accuracy = record.record_metric()
            if accuracy < 0.5:
                accuracy = 0
            else:
                accuracy -= 0.5
            model_accuracy.append(accuracy)
            ##############################################
        total_accuracy = 0
        for acc in model_accuracy:
            total_accuracy += acc
                
        if MODEL_NUM <= 2:
            #######################################################
            ## weight for base model and in-cluster weight,      ##
            ## out-of-cluster weight = 1 - W_base - W_in-cluster ##
            #######################################################
            weights = [[model_accuracy[0] / total_accuracy, model_accuracy[1] / total_accuracy], 
                       [0.1, 0.85], [0.1, 0.6], [0.2, 0.6], [0.3, 0.5], [0.4, 0.4], [0.5, 0.5],
                       [0.6, 0.3], [0.5, 0.35], [0.5, 0.3], [0.4, 0.35], [0.4, 0.3], [0.35, 0.35]]
        else:
            #####################
            ## Weight for mnli ##
            #####################
            weights = [[model_accuracy[0] / total_accuracy, model_accuracy[1] / total_accuracy], 
                       [0.2, 0.5], [0.3, 0.4], [0.4, 0.3], [0.5, 0.2], [0.5, 0.5], [0.25, 0.25]]
        for weight in weights:
            ##############################################
            record.init_validation()
            for batch in valid_dataloader:
                b_input_ids, b_input_attention, b_input_type, b_input_label = batch
                b_input_ids = b_input_ids.type(torch.LongTensor).to(device)
                b_input_attention = b_input_attention.type(torch.LongTensor).to(device)
                b_input_type = b_input_type.type(torch.LongTensor).to(device)
                if "stsb" in INPUT_DIR:
                    b_input_label = b_input_label.type(torch.FloatTensor).to(device)
                else:
                    b_input_label = b_input_label.type(torch.LongTensor).to(device)
                
                with torch.no_grad():
                    outputs = model(input_ids = b_input_ids, 
                                    attention_mask = b_input_attention,         
                                    token_type_ids = b_input_type,
                                    active_cluster = True,
                                    mode = "validation",
                                    weighted = weight)
                
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                label_ids = b_input_label.to('cpu').numpy()
            
                record.get_score(logits, label_ids, config.num_labels)
                    
                del b_input_ids
                del b_input_attention
                del b_input_type
                del b_input_label
                del outputs
                torch.cuda.empty_cache()
            record.record_metric()
            ##############################################
        if "mnli" in TASK_NAME:
            weights = [[1, 0], [0, 1], [0, 0]]
            model_accuracy = []
            for weight in weights:
                ##############################################
                record2.init_validation()
                for batch in valid_mis_dataloader:
                    b_input_ids, b_input_attention, b_input_type, b_input_label = batch
                    b_input_ids = b_input_ids.type(torch.LongTensor).to(device)
                    b_input_attention = b_input_attention.type(torch.LongTensor).to(device)
                    b_input_type = b_input_type.type(torch.LongTensor).to(device)
                    if "stsb" in INPUT_DIR:
                        b_input_label = b_input_label.type(torch.FloatTensor).to(device)
                    else:
                        b_input_label = b_input_label.type(torch.LongTensor).to(device)
                    
                    with torch.no_grad():
                        outputs = model(input_ids = b_input_ids, 
                                        attention_mask = b_input_attention,         
                                        token_type_ids = b_input_type,
                                        active_cluster = True,
                                        mode = "validation",
                                        weighted = weight)
                
                    logits = outputs[0]
                    logits = logits.detach().cpu().numpy()
                    label_ids = b_input_label.to('cpu').numpy()
            
                    record2.get_score(logits, label_ids, config.num_labels)
                    
                    del b_input_ids
                    del b_input_attention
                    del b_input_type
                    del b_input_label
                    del outputs
                    torch.cuda.empty_cache()
                accuracy = record2.record_metric()
                if accuracy < 0.5:
                    accuracy = 0
                else:
                    accuracy -= 0.5
                model_accuracy.append(accuracy)
                ##############################################
            total_accuracy = 0
            for acc in model_accuracy:
                total_accuracy += acc
        
            if MODEL_NUM <= 2:
                weights = [[model_accuracy[0] / total_accuracy, model_accuracy[1] / total_accuracy], 
                           [0.1, 0.85], [0.2, 0.6], [0.3, 0.5], [0.4, 0.4], 
                           [0.5, 0.5], [0.5, 0.35], [0.4, 0.35], [0.35, 0.35]]
            else:
                weights = [[model_accuracy[0] / total_accuracy, model_accuracy[1] / total_accuracy], 
                           [0.2, 0.5], [0.3, 0.4], [0.4, 0.3], [0.5, 0.2], [0.5, 0.5], [0.25, 0.25]]
            
            for weight in weights:
                ##############################################
                record2.init_validation()
                for batch in valid_mis_dataloader:
                    b_input_ids, b_input_attention, b_input_type, b_input_label = batch
                    b_input_ids = b_input_ids.type(torch.LongTensor).to(device)
                    b_input_attention = b_input_attention.type(torch.LongTensor).to(device)
                    b_input_type = b_input_type.type(torch.LongTensor).to(device)
                    if "stsb" in INPUT_DIR:
                        b_input_label = b_input_label.type(torch.FloatTensor).to(device)
                    else:
                        b_input_label = b_input_label.type(torch.LongTensor).to(device)
                
                    with torch.no_grad():
                        outputs = model(input_ids = b_input_ids, 
                                        attention_mask = b_input_attention,         
                                        token_type_ids = b_input_type,
                                        active_cluster = True,
                                        mode = "validation",
                                        weighted = weight)
                
                    logits = outputs[0]
                    logits = logits.detach().cpu().numpy()
                    label_ids = b_input_label.to('cpu').numpy()
            
                    record2.get_score(logits, label_ids, config.num_labels)
                    
                    del b_input_ids
                    del b_input_attention
                    del b_input_type
                    del b_input_label
                    del outputs
                    torch.cuda.empty_cache()
                record2.record_metric()
                ##############################################

    
    if SAVE != -1:    
        torch.save(model, 
                   SAVE_DIR + '/model_final_' + DATA_NAME)
    
    
if __name__ == "__main__":
    run()