from model.ismark_v6_30bit import INRMarkTrainer

# 从中断的 epoch 54 继续训练
INRMarkTrainer("./config/v6_size256_msg64.yaml").train()
