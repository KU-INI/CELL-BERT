"""
For example,

python3 make_data.py \
    --task_name mrpc
"""










from transformers import BertTokenizer
import torch
from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--task_name', type = str, default = None, required=True)
args = parser.parse_args()
TASK_NAME = args.task_name

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def data_set(data, length = 512):
    sentence = [[], []]
    labels = []
    features = []
    for k in data['train'].features:
        if k != 'label' and k != 'idx':
            features.append(k)
    
    if len(features) == 1:
        for i in range(len(data['train'])):
            sentence[0].append(data['train'][i][features[0]].lower())
            labels.append(data['train'][i]['label'])
        train_inputs = tokenizer(sentence[0], 
                                 return_tensors='pt', 
                                 max_length = length, 
                                 truncation=True, 
                                 padding = 'max_length')
    else:
        for i in range(len(data['train'])):
            sentence[0].append(data['train'][i][features[0]].lower())
            sentence[1].append(data['train'][i][features[1]].lower())
            labels.append(data['train'][i]['label'])
        train_inputs = tokenizer(sentence[0], 
                                 sentence[1], 
                                 return_tensors='pt', 
                                 max_length = length, 
                                 truncation=True, 
                                 padding = 'max_length')
    train_inputs['labels'] = torch.FloatTensor([labels]).T
    return train_inputs
    
def valid_set(data, length = 512):    
    sentence = [[], []]
    labels = []
    features = []
    for k in data['train'].features:
        if k != 'label' and k != 'idx':
            features.append(k)
    if len(features) == 1:
        for i in range(len(data['validation'])):
            sentence[0].append(data['validation'][i][features[0]].lower())
            labels.append(data['validation'][i]['label'])
        validation_inputs = tokenizer(sentence[0], 
                                 return_tensors='pt', 
                                 max_length = length, 
                                 truncation=True, 
                                 padding = 'max_length')
    else:
        for i in range(len(data['validation'])):
            sentence[0].append(data['validation'][i][features[0]].lower())
            sentence[1].append(data['validation'][i][features[1]].lower())
            labels.append(data['validation'][i]['label'])
        validation_inputs = tokenizer(sentence[0], 
                                 sentence[1], 
                                 return_tensors='pt', 
                                 max_length = length, 
                                 truncation=True, 
                                 padding = 'max_length')
    validation_inputs['labels'] = torch.FloatTensor([labels]).T
    
    return validation_inputs

def valid_set_mnli_matched(data):    
    sentence = [[], []]
    labels = []
    features = []
    for k in data['train'].features:
        if k != 'label' and k != 'idx':
            features.append(k)
    if len(features) == 1:
        for i in range(len(data['validation_matched'])):
            sentence[0].append(data['validation_matched'][i][features[0]].lower())
            labels.append(data['validation_matched'][i]['label'])
        validation_inputs = tokenizer(sentence[0], 
                                 return_tensors='pt', 
                                 max_length = 128, 
                                 truncation=True, 
                                 padding = 'max_length')
    else:
        for i in range(len(data['validation_matched'])):
            sentence[0].append(data['validation_matched'][i][features[0]].lower())
            sentence[1].append(data['validation_matched'][i][features[1]].lower())
            labels.append(data['validation_matched'][i]['label'])
        validation_inputs = tokenizer(sentence[0], 
                                 sentence[1], 
                                 return_tensors='pt', 
                                 max_length = 128, 
                                 truncation=True, 
                                 padding = 'max_length')
    validation_inputs['labels'] = torch.LongTensor([labels]).T
    
    return validation_inputs

def valid_set_mnli_mismatched(data):    
    sentence = [[], []]
    labels = []
    features = []
    for k in data['train'].features:
        if k != 'label' and k != 'idx':
            features.append(k)
    if len(features) == 1:
        for i in range(len(data['validation_mismatched'])):
            sentence[0].append(data['validation_mismatched'][i][features[0]].lower())
            labels.append(data['validation_mismatched'][i]['label'])
        validation_inputs = tokenizer(sentence[0], 
                                 return_tensors='pt', 
                                 max_length = 128, 
                                 truncation=True, 
                                 padding = 'max_length')
    else:
        for i in range(len(data['validation_mismatched'])):
            sentence[0].append(data['validation_mismatched'][i][features[0]].lower())
            sentence[1].append(data['validation_mismatched'][i][features[1]].lower())
            labels.append(data['validation_mismatched'][i]['label'])
        validation_inputs = tokenizer(sentence[0], 
                                 sentence[1], 
                                 return_tensors='pt', 
                                 max_length = 128, 
                                 truncation=True, 
                                 padding = 'max_length')
    validation_inputs['labels'] = torch.LongTensor([labels]).T
    
    return validation_inputs

if __name__ == "__main__":
    dataset = load_dataset("glue", TASK_NAME)
    if TASK_NAME == "mrpc" or TASK_NAME == "cola" or TASK_NAME == "rte":
        train = data_set(dataset)
        validation = valid_set(dataset)
        torch.save(train, f'./dataset/{TASK_NAME}_train')
        torch.save(validation, f'./dataset/{TASK_NAME}_validation')
    elif TASK_NAME == "mnli":
        length = 128
        train = data_set(dataset, length)
        torch.save(train, './dataset/mnli_train')
        valid_match = valid_set_mnli_matched(dataset)
        torch.save(valid_match, './dataset/mnli_validation_matched')
        valid_mis_match = valid_set_mnli_mismatched(dataset)
        torch.save(valid_mis_match, './dataset/mnli_validation_mismatched')
    else:
        length = 128
        train = data_set(dataset, 128)
        validation = valid_set(dataset, 128)
        torch.save(train, f'./dataset/{TASK_NAME}_train')
        torch.save(validation, f'./dataset/{TASK_NAME}_validation')