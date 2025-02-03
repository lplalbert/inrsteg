import torch
from FastTools.util.TrainUtil import Args
# from model.inr_steg_v4 import INRMark, train
from model.ismark_final_v13 import INRMarkTrainer

# train()

INRMarkTrainer("/home/sn/workspace/inrsteg/config/main.yaml").train()
