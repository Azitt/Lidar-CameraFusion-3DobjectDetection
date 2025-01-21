import tqdm
import time
import numpy as np
import pandas as pd
from functools import partial


import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.nn.utils import clip_grad_norm_

import common_utils, commu_utils
from fastai_optim import OptimWrapper
from learning_schedules_fastai import CosineWarmupLR, OneCycle


def build_optimizer(model, optim_cfg):
    if optim_cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    elif optim_cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM
        )
    elif optim_cfg.OPTIMIZER == 'adam_onecycle':
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )
    else:
        raise NotImplementedError

    return optimizer



def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST]
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * optim_cfg.LR_DECAY
        return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR)

    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs
    if optim_cfg.OPTIMIZER == 'adam_onecycle':
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * len(total_iters_each_epoch),
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            )

    return lr_scheduler, lr_warmup_scheduler


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,tbar, leave_pbar=False):
    
    dataloader_iter = iter(train_loader)
    scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=optim_cfg.get('LOSS_SCALE_FP16', 2.0**16))
    pbar = tqdm.tqdm(total=len(train_loader), leave=leave_pbar, desc='train', dynamic_ncols=True)
    data_time = common_utils.AverageMeter()
    batch_time = common_utils.AverageMeter()
    forward_time = common_utils.AverageMeter()
    losses_m = common_utils.AverageMeter()
    end = time.time()

    for cur_it in range(0, len(train_loader)):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')
        
        data_timer = time.time()
        cur_data_time = data_timer - end
        lr_scheduler.step(cur_it)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            loss, tb_dict, disp_dict = model_func(model, batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        scaler.step(optimizer)
        scaler.update()
 
        cur_forward_time = time.time() - data_timer
        cur_batch_time = time.time() - end
        end = time.time()

        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        batch_size = batch.get('batch_size', None)
            
        data_time.update(avg_data_time)
        forward_time.update(avg_forward_time)
        batch_time.update(avg_batch_time)
        losses_m.update(loss.item() , batch_size)
        
        disp_dict.update({
            'loss': loss.item(), 'lr': cur_lr, 
            'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
            'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})', 
            'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
        })
        
        pbar.update()
        pbar.set_postfix(dict(total_it=cur_it))
        tbar.set_postfix(disp_dict)
            
    pbar.close()
    return losses_m


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                total_epochs, model_name):
    
    min_val_loss = np.Inf    

    with tqdm.trange(0, total_epochs, desc='epochs', dynamic_ncols=True, leave=True) as tbar:
        for cur_epoch in tbar:
            # train one epoch
            loss_m = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=lr_scheduler,
                optim_cfg=optim_cfg,
                tbar=tbar, 
                leave_pbar=(cur_epoch + 1 == total_epochs))

            # if validation loss has decreased, save model and reset variable
            if loss_m.avg <= min_val_loss:
                min_val_loss = loss_m.avg

                # save model and optimizer states
                optim_state = optimizer.state_dict() if optimizer is not None else None
                model_state = model.state_dict()
                state = {'epoch': cur_epoch, 'model_state': model_state, 'optimizer_state': optim_state}
                filename = '{}.pth'.format(model_name)
                torch.save(state, filename, _use_new_zipfile_serialization=False)


    # plot results
    results = pd.DataFrame(results)
    # plot_training_results(results, model_name)
    return results