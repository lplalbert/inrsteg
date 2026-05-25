from abc import abstractmethod
from typing import Iterable
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

# DDP训练教程
# https://github.com/KaiiZhang/DDP-Tutorial/blob/main/DDP-Tutorial.md
# https://www.cnblogs.com/chentiao/p/17666330.html
# https://zhuanlan.zhihu.com/p/98535650
# https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

def _init_ddp_group(rank, world_size, port="65565", backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    pass


def InitProcess(args):
    # 初始化GPU
    if args.gpus is None:
        args.device = 'cpu'
        args.rank = -1
        args.ddp = False
        EasyInit(args)
    elif len(args.gpus) == 1:
        args.device = 'cuda:' + str(args.gpus[0])
        args.rank = -1
        args.ddp = False
        EasyInit(args)
    else:
        # 需要DDP
        args.world_size = len(args.gpus)
        args.ddp = True
        mp.spawn(
            EasyInit,
            args=(args,),
            nprocs=len(args.gpus),
            join=True
        )
        pass
     
    pass

def EasyInit(
    rank, 
    args,
    build_model=None, # 创建模型
    build_dataset=None, # 创建数据集
    ):
    
    if args.ddp:
        _init_ddp_group(rank, args.world_size, args.port)
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_device(rank) # 这里设置 device ，后面可以直接使用 data.cuda(),否则需要指定 rank
        pass
    else:
        device = torch.device(args.device)
        torch.cuda.set_device(device)

    if args.seed is not None:
        # 确定随机种子
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        pass
    
    # 创建model
    model = build_model(args)
    if args.ddp:
        if args.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank])
        pass
    
    # 创建dataset
    datasets = build_dataset(args)
    if isinstance(datasets, Iterable):
        train_dataset, valid_dataset = datasets
    else:
        train_dataset = datasets
        valid_dataset = None
    
    if args.ddp:
        train_sampler = DDP(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            sampler=train_sampler
            )
        
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True
            )
        pass
 
    optimizer = optim.SGD(model.parameters())
 
    for epoch in range(100):
       train_sampler.set_epoch(epoch)
       for batch_idx, (data, target) in enumerate(train_loader):
          data = data.cuda()
          target = target.cuda()
          ...
          output = model(images)
          loss = criterion(output, target)
          ...
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          
    pass


