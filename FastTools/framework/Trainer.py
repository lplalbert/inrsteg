from abc import abstractmethod
import time
from typing import Iterable, List, Type
import lightning as L
import torch
from lightning.fabric.loggers import TensorBoardLogger, CSVLogger
import torch.utils
import torch.utils.data
from FastTools.framework.logger import TerminalLogger
from FastTools.framework.model import EasyModel
import os
from tqdm import tqdm

# 基于Fabric 构建的Trainer
# https://github.com/Lightning-AI/pytorch-lightning/blob/master/examples/fabric/build_your_own_trainer/trainer.py
# https://blog.csdn.net/u011119817/article/details/134036401

class EasyTrainer(object):
    def __init__(self, args):
        self.args = args
        self.log_freq = args.log_freq
        self.gpus = args.gpus
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.val_batch_size = args.val_batch_size
        self.val_num_workers = args.val_num_workers
        self.pin_memory = args.pin_memory
        self.ckpt_shuffix = "ept"
        
        self.terminal_logger = TerminalLogger(args.max_steps)
        
        self.is_multi_gpu = True if (isinstance(args.gpus, int) and args.gpus > 1) or (len(args.gpus) > 1) else False
        
        self.hparams = {
            "step": 0
        }
        
        # 设置保存路径
        time_str = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        self.save_path = os.path.join(args.root_dir, args.name, time_str)

        # 创建logger
        tb_logger = TensorBoardLogger(root_dir=os.path.join(self.save_path, "tensorboard"))
        csv_logger = CSVLogger(root_dir=os.path.join(self.save_path, "csv"))
        
        # print('devices: ', args.gpus)
        # 创建fabric
        self.fabric = L.Fabric(
            devices=self.gpus,
            loggers = [tb_logger, csv_logger],
        )
        pass
    
    def log(self, name, value, log_freq=None, sync_dict=False):
        if log_freq is None:
            log_freq = self.log_freq
        if self.hparams["step"] % log_freq == 0:
            self.fabric.log(name, value)
            self.terminal_logger.log(name, value, sync_dict)
            
        pass
    
    def log_dict(self, dict_value, log_freq=None, sync_dict=False):
        if log_freq is None:
            log_freq = self.log_freq
        if self.hparams["step"] % log_freq == 0:
            self.fabric.log_dict(dict_value)
            self.terminal_logger.log_dict(dict_value, sync_dict)
        pass
    
    
    def is_rank_zero(self):
        """判断是否是主进程
        """
        return self.fabric.global_rank == 0
    
    @abstractmethod
    def build_models_and_optimizers(self, args) -> List:
        """建立模型和对应的优化器列表

        Args:
            args (_type_): 超参数

        Returns:
            list[EasyModel, torch.optim.Optimizer]: 返还列表，第一个元素是模型，其余元素是优化器
        """
        pass
    
    @abstractmethod
    def build_datasets(self, args) -> List[torch.utils.data.Dataset]:
        pass
    

    
    @abstractmethod
    def train_step(self, model, batch):
        
        pass
    
    @abstractmethod
    def valid_step(self, model, batch):
        
        pass
    
    def backward(self, loss: torch.Tensor, retain_graph=False):
        self.fabric.backward(loss, retain_graph=retain_graph)
        pass
    
    def load_ckpt(self):
        args = self.args
        # 加载模型
        if args.ckpt_path.endswith(self.ckpt_shuffix):
            if args.only_load_model:
                self.fabric.load(args.ckpt_path, model=self.model)
            else:
                state = {
                    "model": self.model,
                    "optimizers": self.optimizers,
                    "hparams": self.hparams
                }
                self.fabric.load(args.ckpt_path, state)
        else:
            self.fabric.load_raw(args.ckpt_path, model=self.model)

        pass
    
    def save_ckpt(self, name=None):
        
        args = self.args
        if name is None:
            name = "model_{}.{}".format(self.hparams["step"], self.ckpt_shuffix)
        path = os.path.join(self.save_path, name)
        state = {
            "model": self.model,
            "optimizers": self.optimizers,
            "hparams": self.hparams
        }
        self.fabric.save(os.path.join(self.save_path, path), state)
        
        
        
    
    def run(self):
        #==========================
        # 启动fabric
        #==========================
        self.fabric.launch()
        
        
        args = self.args
        
        #==========================
        # 建立模型和优化器
        #==========================
        self.model, *self.optimizers = self.build_models_and_optimizers(args)
        self.model, *self.optimizers = self.fabric.setup(self.model, *self.optimizers)
        
        #==========================
        # 加载权重
        #==========================
        if args.ckpt_path is not None:
            self.load_ckpt()

        #==========================
        # 建立数据集
        #==========================
        self.datasets = self.build_datasets(args)
        if isinstance(self.datasets, Iterable):
            self.datasets[0] = torch.utils.data.DataLoader(self.datasets[0], batch_size=self.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
            self.datasets[1] = torch.utils.data.DataLoader(self.datasets[1], batch_size=self.val_batch_size, shuffle=False, num_workers=args.val_num_workers, pin_memory=args.pin_memory)
            self.datasets[0] = self.fabric.setup_dataloaders(self.datasets[0], use_distributed_sampler=self.is_multi_gpu)
            self.datasets[1] = self.fabric.setup_dataloaders(self.datasets[1], use_distributed_sampler=self.is_multi_gpu)
        else:
            self.datasets[0] = torch.utils.data.DataLoader(self.datasets[0], batch_size=self.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
            self.datasets[0] = self.fabric.setup_dataloaders(self.datasets[0], use_distributed_sampler=self.is_multi_gpu)
        pass
    
        
        # =========================
        # 训练
        # =========================
        start_time = time.time()
        while self.hparams['step'] < args.max_steps:
            
            for batch in self.datasets[0]:
                self.hparams['step'] += 1
                
                self.model.train()
                self.train_step(self.model, batch)
                
                
                # =======================
                # 打印日志, fabric.print
                # =======================
                if self.hparams['step'] % args.log_freq == 0: 
                    self.terminal_logger.print(self.fabric.print, self.hparams['step'])

                
                if len(self.datasets) > 1 and self.hparams['step'] % args.valid_freq == 0:
                    self.model.eval()
                    for batch in self.datasets[1]:
                        self.valid_step(self.model, batch)
                        pass
                    
                    # =======================
                    # 输出valid统计结果
                    # =======================
                    self.terminal_logger.print(self.fabric.print, self.hparams['step'], True)
                pass
            
                if self.hparams['step'] % args.save_freq == 0:
                    self.save_ckpt()
                    pass
                
                pass
            pass
    
    pass



if __name__ == "__main__":
    
    pass