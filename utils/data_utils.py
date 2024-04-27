import scipy
import numpy as np 
from pathlib import Path
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def process_signal(brain_list, block_list):
    return brain_list

def process_text(arr):
    return [str.strip() for str in arr]

def process_file(data_file):
    
    data = scipy.io.loadmat(data_file)
    date = data_file.stem

    n_trials = data['blockIdx'].shape[0]

    date_list = [date for _ in range(n_trials)]

    brain_list = data['spikePow'][0][:]
    block_list =  data['blockIdx'][:]

    brain_list = process_signal(brain_list, block_list)
    
    sentence_list = data['sentenceText']
    sentence_list = process_text(sentence_list)

    return brain_list, sentence_list, date_list


def process_all_files(path):
    
    data_res = {'brain_list':[], 'sentence_list':[], 'date_list':[]}
    
    for data_file in sorted(path.glob('*.mat')):

        brains, sentences, dates = process_file(data_file)
        
        data_res['brain_list'].extend(brains)
        data_res['sentence_list'].extend(sentences)
        data_res['date_list'].extend(dates)

    return data_res



class BrainDataset(Dataset):
    def __init__(self, path): 

        data = process_all_files(path)
            
        self.max_tokens = 64
        self.inputs = data['brain_list']
        self.targets = data['sentence_list']
        self.date = data['date_list']
        
        all_labels = []

        ## IMPORTANT TO PAD INDEXES WITH -100 values. 
        # for text in self.targets: 
            # labels = self.processor.tokenizer(text, padding="max_length", max_length=self.max_tokens).input_ids
            # labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
            # all_labels.append(labels)
            
        self.targets_tokens = self.targets
        
    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        
        input = self.inputs[idx]
        target = self.targets_tokens
        date = self.date[idx]
                
        return input, target, date