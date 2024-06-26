from dataclasses import dataclass
from pathlib import Path
import math

import safetensors
import torch
from accelerate import Accelerator
from simple_parsing import ArgumentParser
from accelerate.utils import set_seed
import wandb 

@dataclass
class TrainConfig():
    exp_name: str = 'default'

    batch_size: int = 256
    grad_accum: int = 1

    p_augs: float = 0.0

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    max_steps: int = 100_000
    eval_interval: int = 1_000
    
    use_scheduler: bool = True
    warmup_iters: int = 2_000
    lr_decay_iters: int = 50_000
    
    num_workers: int = 3
    pin_memory: bool = True
    
    grad_clip: float = 1.0
    mixed_precision: bool = True

    visualize_predictions: bool = False


def count_parameters(model): 
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    # print in millions
    print(f"Total: {n_total/1e6:.2f}M, Trainable: {n_trainable/1e6:.2f}M")

    return n_total, n_trainable

def init_lr_scheduler(config):
    learning_rate = config.learning_rate
    warmup_iters = config.warmup_iters
    lr_decay_iters = config.lr_decay_iters
    min_lr = learning_rate / 10
    constant_lr = not config.use_scheduler

    def get_lr(it):
        if constant_lr: 
            return learning_rate
        # 1) linear warmup for warmup_iters steps.
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)
    
    print('Completed initialization of scheduler')
    return get_lr

def prepare_data_loaders(train_dataset, val_dataset, config):
    """Prepare the training and validation data loaders."""
    batch_size = config.batch_size // config.grad_accum
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=config.num_workers, 
        pin_memory=config.pin_memory
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=config.num_workers, 
        pin_memory=config.pin_memory
    )
    return train_loader, val_loader

def run_train_model(model, datasets, config, project_name='transformer', save_folder=Path('logs')):
    set_seed(42)

    mp = 'fp16' if config.mixed_precision else 'no'
    accelerator = Accelerator(mixed_precision=mp, 
                                gradient_accumulation_steps=config.grad_accum, 
                                device_placement=True, 
                                split_batches=True, 
                                log_with ='wandb')
    accelerator.init_trackers(
                project_name=project_name, 
                config=config)

    print('Device for training: ', accelerator.device)
    print('Num devices: ', accelerator.num_processes)

    save_folder = save_folder / config.exp_name
    save_folder.mkdir(parents=True, exist_ok=True)

    ## Prepare data, optimizer and scheduler

    train_dataset, val_dataset = datasets
    train_loader, val_loader = prepare_data_loaders(train_dataset, val_dataset, config)

    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=config.learning_rate, 
                                  weight_decay=config.weight_decay)
    scheduler = init_lr_scheduler(config)

    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    
    overall_step = 0
    best_val_loss = float('inf')

    while True:
        for batch in train_loader:
            lr = scheduler(overall_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            with accelerator.accumulate(model):
                optimizer.zero_grad(set_to_none=True)
                
                inputs, labels, date_info = batch

                loss, _ = model(inputs, labels, date_info=date_info)
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_value_(model.parameters(), config.grad_clip)
                optimizer.step()

            overall_step += 1
            accelerator.print('*', end = '')
            accelerator.log({'train/loss': loss.item(), 
                             'lr': lr}, step=overall_step)

            if (overall_step % config.eval_interval) == 0 and accelerator.is_main_process:
                model.eval()
                val_loss_list = []
                for batch in val_loader:
                    inputs, labels, date_info = batch
                    with torch.no_grad():
                        val_loss, _ = model(inputs, labels, date_info)
                    val_loss_list.append(val_loss)
                
                ## printing 
                mean_val_loss = torch.stack(val_loss_list).mean()


                print(f"overall_steps {overall_step}: {loss.item()}")
                print(f"val loss: {mean_val_loss}")
                accelerator.log({'val/loss': mean_val_loss},step=overall_step)
            
                ## saving weights (if better)
                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                
                    save_path = save_folder / f"step_{overall_step}_loss_{mean_val_loss:.4f}.safetensors"
                    safetensors.torch.save_model(model, save_path)
                    print('saved model: ', save_path.name)
                                    
                # ## Visualize 
                # if config.visualize_predictions is True:
                #     if accelerator.is_main_process:
                #         visualize(model, val_loader)
                
                model.train()
            
            if overall_step > config.max_steps:
                accelerator.end_training()
                print('Complete training')
                break



def simple_train_model(model, datasets, config, project_name='transformer'):
    set_seed(42)
    wandb.init(project=project_name)

    SAVE_FOLDER = Path('logs') / config.exp_name
    SAVE_FOLDER.mkdir(parents=True, exist_ok=True)

    ## Prepare data, optimizer and scheduler
    train_dataset, val_dataset = datasets
    train_loader, val_loader = prepare_data_loaders(train_dataset, val_dataset, config)

    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=config.learning_rate, 
                                  weight_decay=config.weight_decay)
    scheduler = init_lr_scheduler(config)
    
    overall_step = 0
    best_val_loss = float('inf')

    # model.train().to('float').to('cuda')

    while True:
        for batch in train_loader:
            lr = scheduler(overall_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                        
            optimizer.zero_grad(set_to_none=True)

            inputs, labels, date_info = batch
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            date_info = date_info.to('cuda')

            loss, _ = model(inputs, labels, date_info=date_info)
            loss.backward()
            optimizer.step()

            overall_step += 1
            print('*', end = '')
            wandb.log({'train/loss': loss.item(), 
                       'lr': lr}, step=overall_step)

            if (overall_step % config.eval_interval) == 0:
                model.eval()
                val_loss_list = []
                for batch in val_loader:
                    inputs, labels, date_info = batch
                    with torch.no_grad():
                        val_loss, _ = model(inputs, labels, date_info)
                    val_loss_list.append(val_loss)
                
                ## printing 
                mean_val_loss = torch.stack(val_loss_list).mean()


                print(f"overall_steps {overall_step}: {loss.item()}")
                print(f"val loss: {mean_val_loss}")
                wandb.log({'val/loss': mean_val_loss},step=overall_step)
            
                ## saving weights (if better)
                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                
                    save_path = SAVE_FOLDER / f"step_{overall_step}_loss_{mean_val_loss:.4f}.safetensors"
                    safetensors.torch.save_model(model, save_path)
                    print('saved model: ', save_path.name)
                
                model.train()
            
            if overall_step > config.max_steps:
                print('Complete training')
                break