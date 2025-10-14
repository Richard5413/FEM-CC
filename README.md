# FEM-CC
Feature Enhancement Module Based on Class-Centric Loss for Fine-Grained Visual Classification

# enviroment 
- pytorch  
- scikit-learn  
- wandb
- numpy
- pandas
- timm: https://github.com/huggingface/pytorch-image-models

# Our pretrained model
Our NAB pretrained model in https://1drv.ms/f/c/ffdce1b3f4e756d5/EjVB1fr8p9FMlVh_McSC5zwBEWZ-6z7NdFui-vaW-NGHyQ

# train
python main.py --c ./configs/TrainCUBSwinT.yaml

# evalation
python main.py --c ./configs/evalCUBswin.yaml

# Cite

```bibtex
@ARTICLE{11197586,
  author={Wang, Daohui and Xinyu, He and Lyu, Shujing and Tian, Wei and Lu, Yue},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Feature Enhancement Module Based on Class-Centric Loss for Fine-Grained Visual Classification}, 
  year={2025},
  volume={},
  number={},
  pages={1-14},
  keywords={Transformers;Feature extraction;Visualization;Annotations;Accuracy;Automobiles;Overfitting;Finite element analysis;Computer architecture;Attention mechanisms;Class center;convolutional neural network (CNN);fine-grained visual classification (FGVC);soft label;Transformer},
  doi={10.1109/TNNLS.2025.3613791}
}
