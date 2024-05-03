import scipy
import numpy as np 
from pathlib import Path
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def process_signal(voltage_list, spikes_list, block_list):
    """
    Preprocess voltages / spike counts based on the
    block-consistent features (z-score)
    """
    
    n_trials = len(block_list)
    trial_indices = np.arange(n_trials)
    
    """ Concatenate spikes and voltages """
    # concatenate spike power and threshold crossings along the channels dimension
    
    brain_concat = np.empty(n_trials, dtype=object)
    for i in range(n_trials):
        brain_concat[i] = np.concatenate([voltage_list[i], spikes_list[i]], axis=1)
    
    """ Block-wise Z-score and smoothing """

    brain_processed = np.empty(n_trials, dtype=object)
    
    for block in np.unique(block_list):
        
        trial_mask = block_list == block
        
        brain_appended = np.concatenate(brain_concat[trial_mask], axis=0)
        
        # get row vectors because channels are 2nd dimension (column-wise means and stds)
        block_mean = brain_appended.mean(axis=0)[None, :] 
        block_std  = brain_appended.std(axis=0)[None, :]
        
        block_std[block_std == 0] = 1
        
        # normalize each trial according to the block mean and std (zscore)
        for trial in trial_indices[trial_mask]:
            brain_processed[trial] = (brain_concat[trial] - block_mean) / block_std
            
            """ Gaussian smoothing over time (in the same loop for efficiency)"""
            # here we don't really care if it's causal (does not look into the future)
            # because we are decoding the whole sentence
            brain_processed[trial] = scipy.ndimage.gaussian_filter1d(brain_processed[trial], sigma=1, axis=0)            
            
    return brain_processed

def process_text(arr):
    return [str.strip() for str in arr]

def process_file(data_file):
    
    data = scipy.io.loadmat(data_file)
    date = data_file.stem

    n_trials = data['blockIdx'].shape[0]

    date_list = [date for _ in range(n_trials)]

    voltage_list = data['spikePow'][0][:]
    spikes_list  = data['tx4'][0][:]
    block_list   = data['blockIdx'][:, 0]

    brain_list = process_signal(voltage_list, spikes_list, block_list)
    
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