# 重构的训练框架

from collections.abc import Iterable
import json
from omegaconf import OmegaConf
import torch
from abc import abstractmethod
from datetime import datetime
import os
from typing import Any
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.utils.data as data
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from FastTools.util.TrainUtil import Args


class EngineModel(L.LightningModule):

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.args.step = 0
        self.history = { } # 临时存储数据
        self.automatic_optimization = False
        pass

    @abstractmethod
    def custom_train_step(self, batch, optimizers, schedulers, batch_idx):
        '''
        训练step
        '''        
        pass
    
    def custom_valid_step(self, batch, batch_idx):
        '''
        验证step
        '''
        pass
    @abstractmethod
    def build_optimizers(self, args):
        '''
        构建优化器，可以返回一个或多个优化器
        '''
        pass    
    
    def build_schedulers(self, optimizers, args):
        '''
        构建学习率调度器
        '''
        return None
        pass

    def clip_gradient(self, opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm"):
        self.clip_gradients(opt, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm)
        pass

    
    def training_step(self, batch, batch_idx):
        self.args.batch_idx = batch_idx
        self.args.step += 1
        optimizers = self.optimizers()
        schedulers = self.lr_schedulers()
        if schedulers is not None and len(schedulers) == 1:
            schedulers = schedulers[0]
        if schedulers is not None and len(schedulers) == 1:
            schedulers = schedulers[0]
        self.custom_train_step(batch, optimizers, schedulers, batch_idx)
        pass
    
    def configure_optimizers(self):
        optimizers = self.build_optimizers(self.args)
        schedulers = self.build_schedulers(optimizers, self.args)
        # 判断是否是可迭代对象，如果不是，用list包裹
        if not isinstance(optimizers, Iterable):
            optimizers = [optimizers]

        if schedulers is not None:
            if not isinstance(schedulers, Iterable):
                schedulers = [schedulers]
        if schedulers is not None:
            return optimizers, schedulers
        return optimizers
        pass
    
    def validation_step(self, batch, batch_idx):
        self.custom_valid_step(batch, batch_idx)
    
    

    
    def loss_backward(self, loss, retain_graph=False):
        """手动调用backward

        Args:
            loss (_type_): _description_
        """
        self.manual_backward(loss, retain_graph=retain_graph)
        pass

    def log_img(self, name, img, n_epoch=10, n_step=None, max_num=8):  
        """保存图片, 如果传入n_step则按照step打印，否则按照epoch打印

        Args:
            name (_type_): 图片名称
            img (_type_): 图片tensor
            step (_type_, optional): 当前step，为None则按照当前epoch计算. Defaults to None.
            n_epoch (int, optional): 多少epoch调用一次. Defaults to 10.
        """
        
        
        if n_step is None and (self.current_epoch + 1) % n_epoch == 0 and self.global_rank == 0:
            img = img.detach().cpu()
            if img.size(0) > max_num:
                img = img[:max_num]
            logger = self.logger.experiment
            logger.add_images("img/{}".format(name), img.cpu(), self.current_epoch)
            pass
        
        if n_step is not None and (self.global_step + 1) % n_step == 0 and self.global_rank == 0:
            img = img.detach().cpu()
            if img.size(0) > max_num:
                img = img[:max_num]
            logger = self.logger.experiment
            logger.add_images("img/{}".format(name), img.cpu(), self.global_step)
            pass
        pass
    
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint['args'] = json.dumps(self.args)
    pass




class EngineTrainer():
    def __init__(self, cfg_path):

        default_cfg_path = os.path.join(
            os.path.split(os.path.realpath(__file__))[0],
            "default_cfg.yaml"
        )
        cfg = Args().load(default_cfg_path)
        custom_cfg = Args().load(cfg_path)
        cfg = Args.merge(cfg, custom_cfg)
    
        # cfg = Args().load_from_yaml(default_cfg_path)
        # custom_cfg = Args().load_from_yaml(cfg_path)
        # cfg = cfg.merge(custom_cfg)


        self.cfg = cfg
        self.ckpt_path = cfg.ckpt_path
        self.precision = cfg.precision  

        # 固定随机种子
        if cfg.seed is not None:
            L.seed_everything(cfg.seed)

        # 如果路径不存在，则创建路径
        self.root_path = os.path.join(cfg.root_dir, cfg.name)
        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path)

        # 设置多少论验证一次
        self.val_every_n_epoch = cfg.n_val_epoch
        if self.val_every_n_epoch == None:
            self.val_every_n_epoch = 1
        
        
        # 构建数据集
        datasets = self.build_dataset(cfg)
        if isinstance(datasets, Iterable):
            dataset, val_dataset = datasets
            self.dataset = torch.utils.data.DataLoader(
                dataset,
                cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=self.build_collate_fn(cfg)
                )
            self.val_dataset = torch.utils.data.DataLoader(
                val_dataset,
                cfg.val_batch_size,
                shuffle=False,
                num_workers=cfg.val_num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=self.build_collate_fn(cfg)
                )
        else:
            dataset = datasets
            self.dataset = torch.utils.data.DataLoader(
                dataset,
                cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=self.build_collate_fn(cfg)
                )
            val_dataset = None
            self.val_dataset = None


        # 构建模型 
        self.model = self.build_model(cfg)
        
        if cfg.ckpt_path is not None:
            ckpt = torch.load(cfg.ckpt_path, map_location="cpu")
            if cfg.resume is False:
                self.model = self.load_checkpoint(self.model, ckpt)
            if cfg.resume is True:
                if ckpt['args'] is not None:
                    self.model.args = Args(json.loads(ckpt["args"]))  # 加载存储的超参数
            self.ckpt_path = None
        
        # 构建callback hooks
        callbacks = []
        checkpoint_callback = self.build_checkpoint_callback()
        if checkpoint_callback is not None:
            callbacks.append(checkpoint_callback)
        earlystopping_callback = self.build_earlystopping_callback()
        if earlystopping_callback is not None:
            callbacks.append(earlystopping_callback)
        

        if cfg.find_unused_parameters == True and len(cfg.gpus) > 1:
            strategy="ddp_find_unused_parameters_true"
        else:
            strategy = "auto"
            # strategy="ddp_find_unused_parameters_False"

        # 看是否同步batch norm
        if cfg.sync_batchnorm is None:
            # 如果cfg.gpus是可迭代对象，并且长度大于1，则同步
            # 否则看是否大于1，则同步
            if isinstance(cfg.gpus, Iterable) and len(cfg.gpus) > 1:
                cfg.sync_batchnorm = True
            elif isinstance(cfg.gpus, int) and cfg.gpus > 1:
                cfg.sync_batchnorm = True
            else:
                cfg.sync_batchnorm = False
            pass

        self.trainer = L.Trainer(
            accelerator="gpu", 
            devices=cfg.gpus,
            default_root_dir=self.root_path,
            enable_checkpointing=True,
            sync_batchnorm=cfg.sync_batchnorm,
            max_epochs=cfg.max_epochs,
            callbacks=callbacks,
            strategy=strategy, # ddp_find_unused_parameters_true 可以避免未使用的参数
            precision= cfg.precision,
            check_val_every_n_epoch=self.val_every_n_epoch,
            log_every_n_steps=cfg.log_every_n_steps
        )
        
        pass
    
    # def load_checkpoint(self, model, ckpt):
    #     model.load_state_dict(ckpt["state_dict"], strict=False)
    #     return model
    #     pass

    def load_checkpoint(self, model, ckpt):
        # 获取模型的当前状态字典
        model_state_dict = model.state_dict()
        checkpoint_state_dict = ckpt["state_dict"]
        
        # 创建新的状态字典，只包含同名且形状匹配的参数
        filtered_state_dict = {}
        loaded_count = 0
        skipped_count = 0
        
        for name, param in checkpoint_state_dict.items():
            # 检查参数是否在当前模型中存在
            if name in model_state_dict:
                # 检查形状是否匹配
                if param.shape == model_state_dict[name].shape:
                    filtered_state_dict[name] = param
                    loaded_count += 1
                else:
                    print(f"跳过参数 {name}: 形状不匹配 (checkpoint: {param.shape}, model: {model_state_dict[name].shape})")
                    skipped_count += 1
            else:
                print(f"跳过参数 {name}: 在当前模型中不存在")
                skipped_count += 1
        
        # 加载过滤后的状态字典
        model.load_state_dict(filtered_state_dict, strict=False)
        
        print(f"加载完成: 成功加载 {loaded_count} 个参数, 跳过 {skipped_count} 个参数")
        return model

    @abstractmethod
    def build_dataset(self, cfg):
        '''
        构建数据集, 返回 训练集 和 验证集(可选)
        '''
        pass
    
    @abstractmethod
    def build_model(self, cfg) -> EngineModel:
        '''
        构建model
        '''
        pass

        
    def build_checkpoint_callback(self):
        '''
        用来自定义保存逻辑
        '''
        callback = ModelCheckpoint(
            save_top_k= 5, # 默认保存最好的5个， 需要保存条件
            monitor='val_loss', # 默认使用总loss保存最好的结果
            filename="ckpt-{epoch:02d}-{loss:.4f}",
            save_last=True,
            save_on_train_epoch_end=True,
            save_weights_only=True
        )
        
        return callback
        pass
    
    def build_earlystopping_callback(self):
        # callback = EarlyStopping(
        #     monitor="val_accuracy", 
        #     min_delta=0.00, 
        #     patience=5, 
        #     verbose=False, 
        #     mode="min"
        #     )
        return None
        pass

    def build_collate_fn(self, cfg):
        """自自定义数据集的collate_fn
        """
        return None

    
    def train(self):
        self.trainer.fit(
            self.model, 
            ckpt_path=self.ckpt_path, 
            train_dataloaders=self.dataset, 
            val_dataloaders=self.val_dataset
        )
        pass
    pass