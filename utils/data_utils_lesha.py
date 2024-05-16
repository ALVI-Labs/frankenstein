import numpy as np
from scipy import signal, stats
import scipy.io
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path
import string
from collections import defaultdict


def process_file(data_file, block_normalize=True, spk_threshold='tx3'):
    
    data = scipy.io.loadmat(data_file)
    date = data_file.stem
    n_trials = data['blockIdx'].shape[0]

    voltage_list  = data['spikePow'][0][:]
    spikes_list   = data[spk_threshold][0][:]
    block_list    = data['blockIdx'][:, 0]
    sentence_list = data['sentenceText']

    if block_normalize:
        voltage_list = z_score_per_block_scaling(voltage_list, block_list)
    
    sentence_list = process_text(sentence_list)

    date_list = [date] * n_trials
    return voltage_list, spikes_list, sentence_list, date_list


def process_all_files(path, block_normalize=True, spk_threshold='tx3'):
    
    data = {'voltage_list':[], 'spike_list':[], 'sentence_list':[], 'date_list':[]}
    for data_file in sorted(path.glob('*.mat')):

        voltages, spikes, sentences, dates = process_file(data_file, block_normalize, spk_threshold)  
        data['voltage_list'].extend(voltages)
        data['spike_list'].extend(spikes)
        data['sentence_list'].extend(sentences)
        data['date_list'].extend(dates)

    return data
    


def z_score_per_block_scaling(brain_list, idx_list):
    """
    Perform block-specific scaling on the input brain data.

    Args:
        brain_list (list): List of brain data arrays, each with shape [Time, 256].
        idx_list (list): List of block indices corresponding to each brain data array.

    Returns:
        list: List of scaled brain data arrays.

    # brain_list = [submit_dataset[i][0] for i in range(4)]
    # idx_list = [0, 0, 100, 100]
    # scaled_brain_list = block_specific_scaling(brain_list, idx_list)
    """

    
    # Group brain indices by block index
    block_idxs = defaultdict(list)
    for i, idx in enumerate(idx_list):
        block_idxs[idx].append(i)

    # Create a scaler for each block
    scalers = {}
    for block_idx, indices in block_idxs.items():
        all_brains_cat = np.concatenate([brain_list[i] for i in indices])
        scaler = StandardScaler().fit(all_brains_cat)
        scalers[block_idx] = scaler

    # Scale each brain data array using the corresponding block scaler
    scaled_brain_list = [scalers[idx].transform(brain) for brain, idx in zip(brain_list, idx_list)]
    return scaled_brain_list


def min_max_per_block_scaling(brain_list, idx_list):
    """
    Perform block-specific scaling on the input brain data.

    Args:
        brain_list (list): List of brain data arrays, each with shape [Time, 256].
        idx_list (list): List of block indices corresponding to each brain data array.

    Returns:
        list: List of scaled brain data arrays.

    # brain_list = [submit_dataset[i][0] for i in range(4)]
    # idx_list = [0, 0, 100, 100]
    # scaled_brain_list = block_specific_scaling(brain_list, idx_list)
    """

    
    # Group brain indices by block index
    block_idxs = defaultdict(list)
    for i, idx in enumerate(idx_list):
        block_idxs[idx].append(i)

    # Create a scaler for each block
    scalers = {}
    for block_idx, indices in block_idxs.items():
        all_brains_cat = np.concatenate([brain_list[i] for i in indices])
        scaler = MinMaxScaler().fit(all_brains_cat)
        scalers[block_idx] = scaler

    # Scale each brain data array using the corresponding block scaler
    scaled_brain_list = [scalers[idx].transform(brain) for brain, idx in zip(brain_list, idx_list)]
    return scaled_brain_list




""" TEXT UTILS """

def process_text(arr):
    return [str.strip() for str in arr]


def process_string(text):
    text = text.lower()
    punctuation = string.punctuation.replace("'", "")
    text = ''.join(char for char in text if char not in punctuation)
    return text


def remove_punctuation(text):
    punctuation = string.punctuation.replace("'", "")
    text = ''.join(char for char in text if char not in punctuation)
    return text


def save_sentences_to_txt(fpath, sentences, string_processing_fn):
    with open(fpath, 'w', encoding="utf-8") as file:
        for sentence in sentences:
            file.write(string_processing_fn(sentence) + "\n")
            
            
def load_sentences_from_txt(fpath):
    with open(fpath, 'r', encoding="utf-8") as file:
        sentences = [line.strip() for line in file.readlines()]
    return sentences