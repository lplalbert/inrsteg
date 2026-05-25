"""
从预训练 checkpoint 微调，使用混合数据集 (document_ds + DIV2K)。

用法:
  CUDA_VISIBLE_DEVICES=0,1 python train_finetune.py
"""

from dataset.Mydataset import MyDataset
from model.ismark_v6_30bit import INRMarkTrainer


class FinetuneTrainer(INRMarkTrainer):

    def build_dataset(self, cfg):
        return MyDataset(cfg, data_len=50000), MyDataset(cfg, data_len=100, valid=True)


if __name__ == "__main__":
    FinetuneTrainer("./config/v6_finetune_doc.yaml").train()
