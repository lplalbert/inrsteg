import torch
from FastTools.util.TrainUtil import Args
# from model.inr_steg_v4 import INRMark, train
from model.ismark_v4 import INRMarkTrainer

# train()

INRMarkTrainer("/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/config/v4.yaml").train()
