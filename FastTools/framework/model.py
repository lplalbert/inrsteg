from abc import abstractmethod
import torch
import torch.nn as nn
import lightning as L
class EasyModel(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        pass
    

    pass
