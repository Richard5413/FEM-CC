# FEM-CC
Feature Enhancement Module Based on Class-Centric Loss for Fine-Grained Visual Classification

# enviroment 
-pytorch  
-scikit-learn  
-wandb  
-timm: https://github.com/huggingface/pytorch-image-models

# Our pretrained model
Our NAB pretrained model in https://1drv.ms/f/c/ffdce1b3f4e756d5/EjVB1fr8p9FMlVh_McSC5zwBEWZ-6z7NdFui-vaW-NGHyQ

# train
python main.py --c ./configs/CUB200_SwinT.yaml

# evalation
python main.py --c ./configs/evalCUBswin.yaml
