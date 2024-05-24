import numpy as np
from scipy import signal, stats
import scipy.io
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch import nn
from torch.utils.data import Dataset

import string
import time
from tqdm import tqdm
from pathlib import Path

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List, Tuple, Union, Dict

from peft import LoraConfig
from peft import LoraModel



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
    sentence_list = [process_string(txt) for txt in sentence_list]

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


def count_parameters(model): 
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    # print in millions
    print(f"Total: {n_total/1e6:.2f}M, Trainable: {n_trainable/1e6:.2f}M")

    return n_total, n_trainable


@dataclass
class ModelAdaptationConfig:
    
    # This set of parameters defines the input convolutional layers configuration (model.model.encoder.conv1 / 2)
    fs_whisper: int = 100       # defines the stride of the second convolution. Effective time for the model.
    inputs_stack: str = "stack" # defines whether to use 2d or 1d convolution. Should be the same as PreprocessConfig.inputs_stack
    n_electrodes: int = 256     # number of ELECTRODES (channels can be n_electrodes x n_features (e.g. spikes + voltage))
    n_features: int = 2         # number of FEATURES (e.g. spikes and voltage)
    max_duration: float = 30.0  # maximal signal duration in seconds. DO NOT TOUCH.
    conv1_time_kernel_size: int = 3  # kernel size for time axis conv1. Will affect padding size
    conv2_time_kernel_size: int = 3  # kernel size for time axis conv2
    conv1_out_channels: int = 1280   # only used if changing the conv2
    conv_layer_idx_stride_2: int = 2 # which conv layer to use for stride 2 convolution. Only applies if fs_whisper = 100

    # These parameters define the learnable layer configuration
    adapt_model: str = "full" # can be full, freeze or lora
    all_module_names: Any = ("conv", "encoder", "adapter", "decoder")
    freeze_modules: Any = ("decoder")
    low_rank_adaptation_modules: Any = ("decoder")
    low_rank_adaptation_targets: Any = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")
    lora_config: Any = None


def configure_input_layers(model, config: ModelAdaptationConfig):

    """1. Set convolutional layers properly"""

    assert config.conv1_time_kernel_size % 2 == 1 and config.conv2_time_kernel_size % 2 == 1, "Time kernel size must be an odd number"
    assert config.inputs_stack in ["stack", "concat"], "Inputs stack can be stack or concat"

    # define strides for handling 3000 or 1500 input lengths
    strides = [1, 1]
    if config.fs_whisper == 100:       
        strides[config.conv_layer_idx_stride_2 - 1] = 2
    else:
        assert config.fs_whisper == 50, "fs_whisper can only be 50 or 100 Hz"

    # define paddings to preserve input length after convolution correctly
    paddings = [1, 1]
    paddings[0] = int((config.conv1_time_kernel_size - 1) / 2)
    paddings[1] = int((config.conv2_time_kernel_size - 1) / 2)


    # if inputs are stacked we have to use 2D convolution
    if config.inputs_stack == "stack":

        example_input = torch.zeros((10, config.n_electrodes, config.n_features, int(config.max_duration * config.fs_whisper)))

        # define custom wrapper over 1d convolution (only if needed) 
        class CustomConv1dWrapper(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
                super(CustomConv1dWrapper, self).__init__()
                self.conv1 = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
                self.stride = stride
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.kernel_size = (kernel_size[1], )
                self.stride = (stride[1], )
                self.padding = (padding[1], )
        
            def forward(self, x):
                x = self.conv1(x)
                return torch.squeeze(x, 2)

        # define conv1 Custom 2d-convolution params (acts like 2D, is seen like 1D)
        in_channels = config.n_electrodes
        out_channels = config.conv1_out_channels
        kernel_size = (config.n_features, config.conv1_time_kernel_size)
        stride = (1, strides[0])
        padding = (0, paddings[0])
        # final result new conv1 layer
        new_conv1 = CustomConv1dWrapper(in_channels, out_channels, kernel_size, stride, padding)


    if config.inputs_stack == "concat":

        example_input = torch.zeros((10, config.n_electrodes * config.n_features, int(config.max_duration * config.fs_whisper)))
        
        # define conv1 1D-convoluation params
        in_channels = config.n_electrodes * config.n_features
        out_channels = config.conv1_out_channels
        kernel_size = (config.conv1_time_kernel_size, )
        stride = (strides[0], )
        padding = (paddings[0], )
        # define new conv1 layer
        new_conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)


    # define conv1 1D-convoluation params
    in_channels = config.conv1_out_channels
    out_channels = model.model.encoder.conv2.out_channels
    kernel_size = (config.conv2_time_kernel_size, )
    stride = (strides[1], )
    padding = (paddings[1], )
    # define new conv2 layer
    new_conv2 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

    # set new layers
    model.model.encoder.conv1 = new_conv1
    model.model.encoder.conv2 = new_conv2

    # test if the inputs are passed correctly
    print("Input size: ", example_input.shape)
    conv1_output = model.model.encoder.conv1(example_input)
    print("Conv1 output size: ", conv1_output.shape)
    conv2_output = model.model.encoder.conv2(conv1_output)
    print("Conv2 output size: ", conv2_output.shape)

    print("\n Number of parameters conv1:")
    print(model.model.encoder.conv1)
    count_parameters(model.model.encoder.conv1)

    print("\nNumber of parameters conv2: ")
    print(model.model.encoder.conv2)
    count_parameters(model.model.encoder.conv2)


    return model


def configure_learnable_layers(model, config: ModelAdaptationConfig):

    assert config.adapt_model in ["full", "freeze", "lora"], "adapt_model can be full, freeze or lora"

    print("Full model parameters: ")
    count_parameters(model)
    print()
    
    """ SETTING UP LORA / FREEZING """
    if config.adapt_model == "full":
        print("Returning full model...")
        return model
        
    if config.adapt_model == "freeze":
        if "conv" in config.freeze_modules:
            print("Freezing full model conv layers...")
            
            for pn, p in model.model.encoder.conv1.named_parameters():
                p.requires_grad = False
            for pn, p in model.model.encoder.conv2.named_parameters():
                p.requires_grad = False
                
        if "encoder" in config.freeze_modules:
            print("Freezing full model encoder layers except conv...")
            
            for pn, p in model.model.encoder.named_parameters():
                if "conv" not in pn:
                    p.requires_grad = False
                    
        if "adapter" in config.freeze_modules: 
            print("Freezing full model adapter layers...")
            
            for pn, p in model.model.decoder.named_parameters():
                if "encoder" in pn:
                    p.requires_grad = False
                    
        if "decoder" in config.freeze_modules:
            print("Freezing full model decoder layers...")
            
            for pn, p in model.model.decoder.named_parameters():
                if "encoder" not in pn:
                    p.requires_grad = False
                    
        if "sanyafreeze" in config.freeze_modules:
            print("Freezing embed and proj_out layers (sanyafreeze)...")
            
            for param in model.model.decoder.embed_tokens.parameters():
                param.requires_grad = False
            
            for param in model.model.decoder.embed_positions.parameters():
                param.requires_grad = False
            
            for param in model.proj_out.parameters():
                param.requires_grad = False

        print("\nParameters after FREEZE: ")
        count_parameters(model)

    if config.adapt_model == "lora":
        assert config.lora_config is not None, "None error: please set the LoraConfig instance first"
        lora_config = config.lora_config
        lora_config.target_modules = config.low_rank_adaptation_targets
        print(lora_config)
        
        if "encoder" in config.low_rank_adaptation_modules:
            print("Putting LORA on encoder...")
            model.model.encoder = LoraModel(model.model.encoder, lora_config, "default")
            
        if "decoder" in config.low_rank_adaptation_modules:
            print("Putting LORA on decoder...")
            model.model.decoder = LoraModel(model.model.decoder, lora_config, "default")

        if "full" in config.low_rank_adaptation_modules:
            model.model = LoraModel(model.model, lora_config, "default")
        
        print("\nParameters after LORA: ")
        count_parameters(model)

    return model


@dataclass
class PreprocessConfig:

    inputs_stack: str = "stack" # stack or concat
    
    voltage_scaler: Any = None
    spike_scaler: Any = None
    sentence_tokenizer: Any = None
    
    eval_filter_freq: float = 25.0
    filter_voltage: bool = True
    filter_spikes: bool = False
    
    fs_orig: int = 50
    fs_whisper: int = 100
    max_duration: float = 30.0  # Maximal signal duration in seconds
    resample_type: str = "fft_resample" # or "interpolate"


@dataclass
class AugmentConfig:
    total_augment_probability: float = 0.7
    time_stretch_probability: float = 0.3
    time_stretch_limits: Tuple[float, float] = (0.9, 1.1)
    voltage_noise_probability: float = 0.5
    voltage_noise_snr_limits: Tuple[float, float] = (20, 30)
    voltage_drift_limits: Tuple[float, float] = (0, 0.1)
    spike_noise_probability: float = 0.0
    spike_noise_per_bin_probability: float = 0.001
    no_filter_probability: float = 0.5
    filter_cutoff_freq_limits: Tuple[float, float] = (15.0, 25.0)
    channel_mask_probability: float = 0.3
    channel_mask_fraction_limits: Tuple[float, float] = (1/256, 16/256) # Fraction of channels to mask
    time_mask_probability: float = 0.3
    time_mask_fraction_limits: Tuple[float, float] = (0.01, 0.05)  # Fraction of time to mask
    random_signal_shift_probability: float = 0.5


class WhisperAugmentDataset(Dataset):
    def __init__(self, 
                 voltage_list: List[np.ndarray], 
                 spike_list: List[np.ndarray], 
                 sentence_list: List[str],
                 preprocess_config: PreprocessConfig, 
                 augment_config: AugmentConfig, 
                 is_eval=False):
        
        self.voltage_list = voltage_list
        self.spike_list = spike_list
        self.sentence_list = sentence_list
        self.preprocess_config = preprocess_config
        self.augment_config = augment_config
        self.is_eval = is_eval
        
        self.max_samples = int(preprocess_config.max_duration * preprocess_config.fs_whisper)

        self.process_data(self.preprocess_config)
        
    def process_data(self, config: PreprocessConfig):
        N = len(self.voltage_list)
        self.N = N
        self.XV_list = [None for _ in range(N)]
        self.XS_list = [None for _ in range(N)]

        self.eval_filter_fn = self.get_lp_filter(config.eval_filter_freq, sample_rate=config.fs_whisper)

        for idx in tqdm(range(N), desc="Processing dataset..."):
            XV = self.voltage_list[idx]
            XS = self.spike_list[idx]
            
            # Scale voltage
            if config.voltage_scaler is not None:
                XV = config.voltage_scaler.transform(XV)
            
            # Scale spikes
            if config.spike_scaler is not None:
                XS = config.spike_scaler.transform(XS)

            XV, XS = XV.T, XS.T
            
            # Resample if original and target sampling freqs are not aligned
            if  config.fs_whisper != config.fs_orig:
                scaling = int(config.fs_whisper / config.fs_orig)
                XV, XS = self.resample_signals(XV, XS, scaling, config.resample_type)

            # Filter if is eval
            if self.is_eval:
                XV = self.eval_filter_fn(XV) if config.filter_voltage else XV
                XS = self.eval_filter_fn(XS) if config.filter_spikes else XS     

            self.XV_list[idx] = XV.astype(np.float32)
            self.XS_list[idx] = XS.astype(np.float32)

        # take example sample and measure its processing speed
        n_perf_samples = 10
        rand_sample_indices = np.random.permutation(N)[:n_perf_samples]
        t0 = time.perf_counter()
        for idx in rand_sample_indices:
            _ = self[idx]
        t1 = time.perf_counter()
        print(f"Input processing time ~ {(t1 - t0) / n_perf_samples * 1000:.1f} ms")

        # free up some memory
        self.voltage_list = None
        self.spike_list = None

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        
        XV = self.XV_list[idx].copy()
        XS = self.XS_list[idx].copy()

        if not self.is_eval:
            XV, XS, do_augment = self.augment(XV, XS, self.augment_config)
        else:
            do_augment = False

        XV, XS = self.pad_shift(XV, XS, do_augment, self.augment_config)

        # Stack voltage and spike data
        which_stack = self.preprocess_config.inputs_stack
        if which_stack == "stack":
            input_features = np.stack((XV, XS), axis=1)  # Shape: (256, 2, times)
        if which_stack == "concat":
            input_features = np.concatenate((XV, XS), axis=0)

        # Process (tokenize) the sentence
        sentence = self.sentence_list[idx]
        labels = self.preprocess_config.sentence_tokenizer(sentence, return_tensors="pt").input_ids.squeeze()

        return {
            "input_features": torch.tensor(input_features, dtype=torch.float32),
            "labels": labels,
        }

    def augment(self, XV: np.ndarray, XS: np.ndarray, config: AugmentConfig) -> Tuple[np.ndarray, np.ndarray]:

        # set if prior to everything so that    
        do_augment = np.random.rand() <= config.total_augment_probability
        do_filter_augment = np.random.rand() <= config.no_filter_probability

        # apply standart eval filter if not doing filter augmentation or doing any augmentation
        if not do_filter_augment or not do_augment:
            XV = self.eval_filter_fn(XV) if self.preprocess_config.filter_voltage else XV
            XS = self.eval_filter_fn(XS) if self.preprocess_config.filter_spikes else XS     
        
        if not do_augment:
            return XV, XS, do_augment

        # Filter noise
        if do_filter_augment:
            cutoff_freq = np.random.uniform(*config.filter_cutoff_freq_limits)
            filter_fn = self.get_lp_filter(cutoff_freq, sample_rate=self.preprocess_config.fs_whisper)
            XV = filter_fn(XV) if self.preprocess_config.filter_voltage else XV
            XS = filter_fn(XS) if self.preprocess_config.filter_spikes else XS

        # Time stretch
        if np.random.rand() < config.time_stretch_probability:
            stretch_factor = np.random.uniform(*config.time_stretch_limits)
            XV = self.time_stretch(XV, stretch_factor)
            XS = self.time_stretch(XS, stretch_factor)

        # Voltage noise
        if np.random.rand() < config.voltage_noise_probability:
            snr = np.random.uniform(*config.voltage_noise_snr_limits)
            voltage_drift = np.random.uniform(*config.voltage_drift_limits)
            XV = self.add_voltage_noise(XV, snr, voltage_drift)

        # Spike noise
        if np.random.rand() < config.spike_noise_probability:
            XS = self.add_spike_noise(XS, config.spike_noise_per_bin_probability)

        # Channel mask
        if np.random.rand() < config.channel_mask_probability:
            n_chan = XV.shape[0]
            chan_mask_fraction = np.random.uniform(*config.channel_mask_fraction_limits)
            bad_chan_mask = (np.random.binomial(n=1, p=chan_mask_fraction, size=n_chan)).astype(bool)
            XV[bad_chan_mask, :] = 0
            XS[bad_chan_mask, :] = -1

        # Time mask
        if np.random.rand() < config.time_mask_probability:
            n_samples = XV.shape[1]
            time_mask_fraction = np.random.uniform(*config.time_mask_fraction_limits)
            mask_length = int(time_mask_fraction * n_samples)
            start_idx = np.random.randint(0, n_samples - mask_length)
            XV[:, start_idx:start_idx + mask_length] = 0
            XS[:, start_idx:start_idx + mask_length] = -1

        return XV, XS, do_augment


    def pad_shift(self, XV: np.ndarray, XS: np.ndarray, do_augment: bool, config: AugmentConfig) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = XV.shape[1]

        # Pad to max duration (any case)
        XV = np.pad(XV, [(0, 0), (0, max(0, self.max_samples - n_samples))], mode='constant')
        XS = np.pad(XS, [(0, 0), (0, max(0, self.max_samples - n_samples))], mode='constant')    
        
        # Random signal shift if train and coin flip
        if not self.is_eval and do_augment and np.random.rand() < config.random_signal_shift_probability:
            shift_length = np.random.randint(self.max_samples - n_samples)
            XV = np.roll(XV, shift_length, axis=1)
            XS = np.roll(XS, shift_length, axis=1)

        return XV, XS

    def resample_signals(self, XV: np.ndarray, XS:np.ndarray, scaling: int, resample_type=str):
        n_samples = XV.shape[1]
        
        if resample_type == "fft_resample":
            XV = scipy.signal.resample(XV, int(n_samples * scaling), axis=-1)
            XS = scipy.signal.resample(XS, int(n_samples * scaling), axis=-1)

        if resample_type == "interpolate":
            x_old = np.arange(int(n_samples * scaling), step=scaling)
            x_new = np.arange(int((n_samples - 1) * scaling + 1), step=1)
            
            fV = scipy.interpolate.interp1d(x_old, XV, axis=-1, kind='linear', bounds_error=False, fill_value=0)
            fS = scipy.interpolate.interp1d(x_old, XS, axis=-1, kind='linear', bounds_error=False, fill_value=-1)

            XV = fV(x_new)
            XS = fS(x_new)

        return XV, XS

    def time_stretch(self, signal: np.ndarray, stretch_factor: float) -> np.ndarray:
        stretched_signal = scipy.signal.resample(signal, int(signal.shape[1] * stretch_factor), axis=-1)
        return stretched_signal

    def add_voltage_noise(self, signal: np.ndarray, snr: float, voltage_drift: float) -> np.ndarray:
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr / 10))
        noise = np.random.normal(loc=voltage_drift, scale=np.sqrt(noise_power), size=signal.shape)
        return signal + noise

    def add_spike_noise(self, spikes: np.ndarray, per_bin_probability: float) -> np.ndarray:
        config = self.preprocess_config
        original_spikes = config.spike_scaler.inverse_transform(spikes.T).T  # Inverse transform
        noise = np.random.binomial(1, per_bin_probability, size=original_spikes.shape)
        noisy_spikes = original_spikes + noise
        noisy_spikes = config.spike_scaler.transform(noisy_spikes.T).T  # Transform back
        return noisy_spikes

    
    def get_lp_filter(self, cutoff_freq, sample_rate=50):
        if cutoff_freq == 0:
            return lambda x: x
        b, a = signal.butter(4, 2 * cutoff_freq / sample_rate, btype='lowpass')
        return lambda x: signal.filtfilt(b, a, x, padlen=50, axis=-1)



@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    
    feature_extractor: Any
    tokenizer: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # first treat the BRAIN INPUTS (already preprocessed)
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.feature_extractor.pad(input_features, return_tensors="pt")
        # batch = input_features

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

class WhisperStackBrainDataset(Dataset):
    
    def __init__(self, 
                 voltage_list, 
                 spike_list, 
                 sentence_list,
                 tokenizer, 
                 voltage_scaler=None, 
                 spike_scaler=None,
                 augment_train=None, 
                 augment_eval=None,
                 voltage_noise_augment=None,
                 spike_noise_augment=None,
                 noise_probability=0.5,
                 shift_probability=0.5,
                 bad_chan_probability=0.5,
                 bad_chan_fraction=0.05,
                 fs_orig=50, 
                 fs_whisper=100, 
                 max_duration=30, 
                 is_eval=False):
        
        self.voltage_list = voltage_list
        self.spike_list = spike_list
        self.sentence_list = sentence_list
        
        self.tokenizer = tokenizer
        self.voltage_scaler = voltage_scaler
        self.spike_scaler = spike_scaler

        self.noise_probability = noise_probability
        self.voltage_noise_augment = voltage_noise_augment
        self.spike_noise_augment = spike_noise_augment

        self.augment_train = augment_train
        self.augment_eval = augment_eval

        self.shift_probability = shift_probability
        self.bad_chan_probability = bad_chan_probability
        self.bad_chan_fraction = bad_chan_fraction

        
        self.fs_orig = fs_orig
        self.fs_whisper = fs_whisper
        self.max_duration = max_duration
        
        self.is_eval = is_eval

        # dataset preprocessing before __getitem__
        N = len(voltage_list)
        self.N = N
        self.XV_list = [None for _ in range(N)]
        self.XS_list = [None for _ in range(N)]

        for idx in tqdm(range(N), desc="Scaling, resampling the dataset..."):
            
            # 1. scale voltage
            XV = voltage_list[idx]
            XS = spike_list[idx]
            
            if self.voltage_scaler is None:
                XV_scaled = XV
            else:
                flag = int(self.voltage_scaler.__class__.__name__ == "QuantileTransformer")
                XV_scaled = (self.voltage_scaler.transform(XV) * (flag + 1) - flag).astype(np.float32)

            if self.spike_scaler is None:
                XS_scaled = XS
            else:
                XS_scaled = self.spike_scaler.transform(XS).astype(np.float32)

            # 2. Resample from original sampling rate to desired one
            scaling = int(self.fs_whisper / self.fs_orig)
            sig_length = XV_scaled.shape[0]
            # transpose to have the time dimension as the last one
            XV_resampled = scipy.signal.resample(XV_scaled.T, sig_length * scaling, axis=-1)
            XS_resampled = scipy.signal.resample(XS_scaled.T, sig_length * scaling, axis=-1)

            self.XV_list[idx] = XV_resampled
            self.XS_list[idx] = XS_resampled

        # take example sample and measure its processing speed
        n_perf_samples = 10
        rand_sample_indices = np.random.permutation(N)[:n_perf_samples]
        t0 = time.perf_counter()
        for idx in rand_sample_indices:
            _ = self[idx]
        t1 = time.perf_counter()
        print(f"Input processing time ~ {(t1 - t0) / n_perf_samples * 1000:.1f} ms")


    def __len__(self):
        return self.N

    def __getitem__(self, idx):

        XV = self.XV_list[idx]
        XS = self.XS_list[idx]

        n_chan, n_samples = XV.shape

        # Random electrode zeroing out if a coin flip is 1
        p = self.bad_chan_probability
        if not self.is_eval and np.random.choice([0, 1], p=[1-p, p]):
            bad_chan_mask = (np.random.binomial(n=1, p=self.bad_chan_fraction, size=n_chan)).astype(bool)
            XV[bad_chan_mask, :] = 0
            XS[bad_chan_mask, :] = -1

        # Apply audio augmentations if necessary
        augment = self.augment_eval if self.is_eval else self.augment_train
        
        # Pad the signal before augmentation to avoid edge effects
        # pad_length = int(0.2 * n_samples)  # Pad 20% of the signal length on both sides
        X_concat = np.concatenate((XV, XS), axis=0)  # get 512 channels to augment them equally!
        # X_concat_padded = np.pad(X_concat, [(0, 0), (pad_length, pad_length)], mode='reflect')
        try:
            X_aug = X_concat if augment is None else augment(X_concat, sample_rate=self.fs_whisper)
        except:
            X_aug = X_concat
        # X_aug = X_aug_padded[:, pad_length:-pad_length] # Trim the signal back to the original length after augmentation

        # separate back (lazy to think about it now)
        XV_aug = X_aug[:n_chan, :]
        XS_aug = X_aug[n_chan:, :]

        # Noise augment
        p = self.noise_probability
        if not self.is_eval and np.random.choice([0, 1], p=[1-p, p]):
            XV_aug = XV_aug if self.voltage_noise_augment is None else self.voltage_noise_augment(XV_aug, sample_rate=self.fs_whisper)
            XS_aug = XS_aug if self.spike_noise_augment is None else self.spike_noise_augment(XS_aug, sample_rate=self.fs_whisper) 


        n_samples = XV_aug.shape[1]
        
        # Random pad to a desired input length (3000 in case of Whisper)    
        max_samples = int(self.max_duration * self.fs_whisper)
        max_shift = max_samples - n_samples

        # Random pad augmentation
        rand_shift = 0
        p = self.shift_probability
        if not self.is_eval and np.random.choice([0, 1], p=[1-p, p]):
            rand_shift = np.random.randint(max_shift)

        XV_pad = np.pad(XV_aug, [(0, 0), (rand_shift, max_samples - n_samples - rand_shift)])
        XS_pad = np.pad(XS_aug, [(0, 0), (rand_shift, max_samples - n_samples - rand_shift)])

        # Stack voltage and spike data
        input_features = np.stack((XV_pad, XS_pad), axis=1)  # Shape: (256, 2, times)

        # Process (tokenize) the sentence
        sentence = self.sentence_list[idx]
        labels = self.tokenizer(sentence, return_tensors="pt").input_ids.squeeze()

        return {
            "input_features": torch.tensor(input_features),
            "labels": labels,
        }
        



class WhisperSuperBrainDataset(Dataset):
    
    def __init__(self, 
                 voltage_list, 
                 spike_list, 
                 sentence_list,
                 tokenizer, 
                 voltage_scaler=None, 
                 spike_scaler=None,
                 augment_train=None, 
                 augment_eval=None,
                 shift_probability=0.5,
                 fs_orig=50, 
                 fs_whisper=100, 
                 max_duration=30, 
                 is_eval=False):
        
        self.voltage_list = voltage_list
        self.spike_list = spike_list
        self.sentence_list = sentence_list
        
        self.tokenizer = tokenizer
        self.voltage_scaler = voltage_scaler
        self.spike_scaler = spike_scaler

        self.augment_train = augment_train
        self.augment_eval = augment_eval
        self.fs_orig = fs_orig
        self.fs_whisper = fs_whisper
        self.max_duration = max_duration
        self.is_eval = is_eval
        self.shift_probability=shift_probability

        # dataset preprocessing before __getitem__
        N = len(voltage_list)
        X_list = [None for _ in range(N)]

        for idx in tqdm(range(N), desc="Scaling, concatenating, resampling the dataset..."):
            
            # 1. scale voltage
            XV = voltage_list[idx]
            XS = spike_list[idx]
            
            if self.voltage_scaler is None:
                XV_scaled = XV
            else:
                flag = int(self.voltage_scaler.__class__.__name__ == "QuantileTrasformer")
                XV_scaled = (self.voltage_scaler.transform(XV) * (flag + 1) - flag).astype(np.float32)

            # 2. scale spikes
            if self.spike_scaler is None:
                XS_scaled = XS
            else:
                XS_scaled = self.spike_scaler.transform(XS).astype(np.float32)

            # 2. concatenate and transpose
            # interleave
            X_concat = np.empty((XS_scaled.shape[0], XS_scaled.shape[1] * 2)) # 2 * 256 x times
            X_concat[:, 0::2] = XV_scaled
            X_concat[:, 1::2] = XS_scaled
            X_concat = X_concat.T  # channels x times
            
            # 3. resample from orig sampling rate to desired one
            scaling = int(self.fs_whisper / self.fs_orig)
            sig_length = X_concat.shape[1]
            X_resampled = scipy.signal.resample(X_concat, sig_length * scaling, axis=1)

        
            X_list[idx] = X_resampled

        # save data list as an attribute
        self.X_list = X_list

    
    def __len__(self):
        return len(self.voltage_list)

    
    def __getitem__(self, idx):

        X = self.X_list[idx]

        # apply augmentation if necessary (not None)
        augment = self.augment_eval if self.is_eval else self.augment_train
        X_aug = X if augment is None else augment(X, sample_rate=self.fs_whisper)

        # pad to a desired input length (3000 in case of Whisper)
        n_samples = X_aug.shape[1]
        max_samples = int(self.max_duration * self.fs_whisper)
        max_shift = max_samples - n_samples

        # random pad augmentation
        if self.is_eval:
            rand_shift = 0
        else:
            p = self.shift_probability
            rand_shift = np.random.randint(max_shift) * np.random.choice([0, 1], p=[1-p, p])
            
        X_pad = np.pad(X_aug, [(0, 0), (rand_shift, max_samples - n_samples - rand_shift)])
        
        input_features = X_pad

        # process (tokenize) the sentence
        sentence = self.sentence_list[idx]
        labels = self.tokenizer(sentence, return_tensors="pt").input_ids.squeeze()

        return {
            "input_features": torch.tensor(input_features),
            "labels": labels,
        }


def count_parameters(model): 
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    # print in millions
    print(f"Total: {n_total/1e6:.2f}M, Trainable: {n_trainable/1e6:.2f}M")

    return n_total, n_trainable
        


""" TEXT UTILS """

def process_text(arr):
    return [s.strip() for s in arr]


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




import json
from dataclasses import asdict, is_dataclass
from typing import Any

def save_config(config: Any, filename: str) -> None:
    """
    Save a dataclass configuration to a JSON file.

    Args:
    config (Any): The dataclass configuration instance to save.
    filename (str): The path to the file where the configuration will be saved.
    """
    if not is_dataclass(config):
        raise ValueError("config must be a dataclass instance.")
    
    with open(filename, 'w') as f:
        json.dump(asdict(config), f, indent=4)


def load_config(cls: Any, filename: str) -> Any:
    """
    Load a dataclass configuration from a JSON file.

    Args:
    cls (Any): The dataclass type to load.
    filename (str): The path to the file from where the configuration will be loaded.

    Returns:
    Any: The loaded dataclass instance.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    return cls(**data)
