import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Union
import copy

class GCNCombiner(nn.Module):

    def __init__(self, 
                 total_num_selects: int,
                 num_classes: int, 
                 inputs: Union[dict, None] = None, 
                 proj_size: Union[int, None] = None,
                 fpn_size: Union[int, None] = None):
        """
        If building backbone without FPN, set fpn_size to None and MUST give 
        'inputs' and 'proj_size', the reason of these setting is to constrain the 
        dimension of graph convolutional network input.
        """
        super(GCNCombiner, self).__init__()

        assert inputs is not None or fpn_size is not None, \
            "To build GCN combiner, you must give one features dimension."

        ### auto-proj
        self.fpn_size = fpn_size
        if fpn_size is None:
            for name in inputs:
                if len(name) == 4:
                    in_size = inputs[name].size(1)
                elif len(name) == 3:
                    in_size = inputs[name].size(2)
                else:
                    raise ValusError("The size of output dimension of previous must be 3 or 4.")
                m = nn.Sequential(
                    nn.Linear(in_size, proj_size),
                    nn.ReLU(),
                    nn.Linear(proj_size, proj_size)
                )
                self.add_module("proj_"+name, m)
            self.proj_size = proj_size
        else:
            self.proj_size = fpn_size

        ### build one layer structure (with adaptive module)
        num_joints = total_num_selects // 32

        self.param_pool0 = nn.Linear(total_num_selects, num_joints)
        
        A = torch.eye(num_joints)/100 + 1/100
        self.adj1 = nn.Parameter(copy.deepcopy(A))
        self.conv1 = nn.Conv1d(self.proj_size, self.proj_size, 1)
        self.batch_norm1 = nn.BatchNorm1d(self.proj_size)
        
        self.conv_q1 = nn.Conv1d(self.proj_size, self.proj_size//4, 1)
        self.conv_k1 = nn.Conv1d(self.proj_size, self.proj_size//4, 1)
        self.alpha1 = nn.Parameter(torch.zeros(1))

        ### merge information
        self.param_pool1 = nn.Linear(num_joints, 1)
        
        #### class predict
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.proj_size, num_classes)

        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        """
        hs = []
        for name in x:
            if self.fpn_size is None:
                hs.append(getattr(self, "proj_"+name)(x[name]))
            else:
                hs.append(x[name])
        hs = torch.cat(hs, dim=1).transpose(1, 2).contiguous() # B, S', C --> B, C, S
        hs = self.param_pool0(hs)
        ### adaptive adjacency
        q1 = self.conv_q1(hs).mean(1)
        k1 = self.conv_k1(hs).mean(1)
        A1 = self.tanh(q1.unsqueeze(-1) - k1.unsqueeze(1))
        A1 = self.adj1 + A1 * self.alpha1
        ### graph convolution
        hs = self.conv1(hs)
        hs = torch.matmul(hs, A1)
        hs = self.batch_norm1(hs)
        # ds = hs
        ### predict
        hs = self.param_pool1(hs)
        ds = hs
        hs = self.dropout(hs)
        hs = hs.flatten(1)
        hs = self.classifier(hs)

        return hs ,ds

# 单层GCNdotproduct
class GCNCombiner1(nn.Module):
    def __init__(self, 
                 total_num_selects: int,
                 num_classes: int, 
                 inputs: Union[dict, None] = None, 
                 proj_size: Union[int, None] = None,
                 fpn_size: Union[int, None] = None):
        """
        If building backbone without FPN, set fpn_size to None and MUST give 
        'inputs' and 'proj_size', the reason of these setting is to constrain the 
        dimension of graph convolutional network input.
        """
        super(GCNCombiner1, self).__init__()

        assert inputs is not None or fpn_size is not None, \
            "To build GCN combiner, you must give one features dimension."

        # Auto projection for inputs (if no FPN)
        self.fpn_size = fpn_size
        if fpn_size is None:
            for name in inputs:
                if len(inputs[name].size()) == 4:
                    in_size = inputs[name].size(1)
                elif len(inputs[name].size()) == 3:
                    in_size = inputs[name].size(2)
                else:
                    raise ValueError("The size of output dimension of previous must be 3 or 4.")
                m = nn.Sequential(
                    nn.Linear(in_size, proj_size),
                    nn.ReLU(),
                    nn.Linear(proj_size, proj_size)
                )
                self.add_module("proj_" + name, m)
            self.proj_size = proj_size
        else:
            self.proj_size = fpn_size

        # Determine number of graph nodes
        num_joints = total_num_selects // 32
        self.num_joints = num_joints

        # Pool original features to graph nodes
        self.param_pool0 = nn.Linear(total_num_selects, num_joints)

        # Optional: learnable residual adjacency (initialize as weak identity)
        A = torch.eye(num_joints) / 100 + 1 / 100
        # A = torch.eye(num_joints) / 100
        self.adj1 = nn.Parameter(copy.deepcopy(A))  # shape [N, N]

        # GCN layer
        self.conv1 = nn.Conv1d(self.proj_size, self.proj_size, 1)
        self.batch_norm1 = nn.BatchNorm1d(self.proj_size)

        # Graph readout
        self.param_pool1 = nn.Linear(num_joints, 1)

        # Classifier
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.proj_size, num_classes)

    def forward(self, x):
        """
        x: dict of features (B, C, S) or (B, S, C) depending on source
        """
        hs = []
        for name in x:
            if self.fpn_size is None:
                hs.append(getattr(self, "proj_" + name)(x[name]))
            else:
                hs.append(x[name])
        hs = torch.cat(hs, dim=1).transpose(1, 2).contiguous()  # B, S, C -> B, C, S -> B, S, C
        hs = self.param_pool0(hs)  # B, C, N

        # Graph construction via scaled dot-product similarity
        node_feats = hs.transpose(1, 2)  # B, N, C
        attn_scores = torch.matmul(node_feats, node_feats.transpose(1, 2))  # B, N, N
        attn_scores = attn_scores / (node_feats.size(-1) ** 0.5)
        A1 = torch.softmax(attn_scores, dim=-1)  # softmax over last dim

        # Optional: residual adjacency
        A1 = A1 + self.adj1.unsqueeze(0)  # [B, N, N]

        # Graph convolution
        hs = self.conv1(hs)  # B, C, N
        hs = torch.matmul(hs, A1)  # B, C, N
        hs = self.batch_norm1(hs)
        # hs = torch.relu(hs)

        # Predict
        hs = self.param_pool1(hs)  # B, C, 1
        ds = hs
        hs = self.dropout(hs)
        hs = hs.flatten(1)
        hs = self.classifier(hs)

        return hs, ds
        
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):  # x: (B, C, N)
        B, C, N = x.shape
        x = x.permute(0, 2, 1)  # (B, N, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out).permute(0, 2, 1)

class GCNBlock(nn.Module):
    def __init__(self, dim, num_joints):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.conv1 = nn.Conv1d(dim, dim, 1)
        self.conv2 = nn.Conv1d(dim, dim, 1)

        self.mhsa = MultiHeadAttention(dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.1)

        self.alpha = nn.Parameter(torch.zeros(1))
        self.adj = nn.Parameter(torch.eye(num_joints)/100 + 1/100)

        self.conv_q = nn.Conv1d(dim, dim//4, 1)
        self.conv_k = nn.Conv1d(dim, dim//4, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):  # x: (B, C, N)
        residual = x

        # 自适应邻接矩阵
        q = self.conv_q(x).mean(1)  # (B, d_q)
        k = self.conv_k(x).mean(1)
        A = self.tanh(q.unsqueeze(-1) - k.unsqueeze(1))
        A = self.adj + A * self.alpha

        # 第一层 GCN
        x = self.conv1(x)
        x = torch.matmul(x, A)
        x = self.gelu(x)
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)

        # 第二层 GCN
        x = self.conv2(x)
        x = torch.matmul(x, A)
        x = self.gelu(x)
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)

        # 多头注意力
        x = self.mhsa(x)
        x = self.norm3(x.transpose(1, 2)).transpose(1, 2)
        x = x + residual  # 跳跃连接
        x = self.dropout(x)

        return x

class GCNCombiner2(nn.Module):
    def __init__(self, total_num_selects, num_classes, inputs=None, proj_size=None, fpn_size=None):
        super(GCNCombiner, self).__init__()
        assert inputs is not None or fpn_size is not None

        self.fpn_size = fpn_size
        if fpn_size is None:
            for name in inputs:
                if len(inputs[name].size()) == 4:
                    in_size = inputs[name].size(1)
                elif len(inputs[name].size()) == 3:
                    in_size = inputs[name].size(2)
                else:
                    raise ValueError("Input tensor must be 3D or 4D.")
                self.add_module(f"proj_{name}", nn.Sequential(
                    nn.Linear(in_size, proj_size),
                    nn.ReLU(),
                    nn.Linear(proj_size, proj_size)
                ))
            self.proj_size = proj_size
        else:
            self.proj_size = fpn_size

        self.num_joints = total_num_selects // 32
        self.param_pool0 = nn.Linear(total_num_selects, self.num_joints)

        # 三个 GCNBlock，每个包含两层 GCN 和注意力
        self.gcn1 = GCNBlock(self.proj_size, self.num_joints)
        self.gcn2 = GCNBlock(self.proj_size, self.num_joints)
        self.gcn3 = GCNBlock(self.proj_size, self.num_joints)

        self.param_pool1 = nn.Linear(self.num_joints, 1)
        self.classifier = nn.Linear(self.proj_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        hs = []
        for name in x:
            if self.fpn_size is None:
                hs.append(getattr(self, f"proj_{name}")(x[name]))
            else:
                hs.append(x[name])
        hs = torch.cat(hs, dim=1).transpose(1, 2)  # B, C, S -> B, S, C -> B, C, N
        hs = self.param_pool0(hs)

        hs = self.gcn1(hs)
        hs = self.gcn2(hs)
        hs = self.gcn3(hs)

        ds = self.param_pool1(hs)
        out = self.dropout(ds)
        out = out.flatten(1)
        out = self.classifier(out)

        return out, ds

# class GCNCombiner(nn.Module):

#     def __init__(self, 
#                  total_num_selects: int,
#                  num_classes: int, 
#                  inputs: Union[dict, None] = None, 
#                  proj_size: Union[int, None] = None,
#                  fpn_size: Union[int, None] = None):
#         """
#         If building backbone without FPN, set fpn_size to None and MUST give 
#         'inputs' and 'proj_size', the reason of these setting is to constrain the 
#         dimension of graph convolutional network input.
#         """
#         super(GCNCombiner, self).__init__()

#         assert inputs is not None or fpn_size is not None, \
#             "To build GCN combiner, you must give one features dimension."

#         ### auto-proj
#         self.fpn_size = fpn_size
#         if fpn_size is None:
#             for name in inputs:
#                 if len(name) == 4:
#                     in_size = inputs[name].size(1)
#                 elif len(name) == 3:
#                     in_size = inputs[name].size(2)
#                 else:
#                     raise ValusError("The size of output dimension of previous must be 3 or 4.")
#                 m = nn.Sequential(
#                     nn.Linear(in_size, proj_size),
#                     nn.ReLU(),
#                     nn.Linear(proj_size, proj_size)
#                 )
#                 self.add_module("proj_"+name, m)
#             self.proj_size = proj_size
#         else:
#             self.proj_size = fpn_size

#         ### build one layer structure (with adaptive module)
#         num_joints = total_num_selects // 32

#         self.param_pool0 = nn.Linear(total_num_selects, num_joints)
        
#         A = torch.eye(num_joints)/100 + 1/100
#         self.adj1 = nn.Parameter(copy.deepcopy(A))
#         self.conv1 = nn.Conv1d(self.proj_size, self.proj_size, 1)
#         self.batch_norm1 = nn.BatchNorm1d(self.proj_size)
        
#         self.conv_q1 = nn.Conv1d(self.proj_size, self.proj_size//4, 1)
#         self.conv_k1 = nn.Conv1d(self.proj_size, self.proj_size//4, 1)
#         self.alpha1 = nn.Parameter(torch.zeros(1))

#         ### merge information
#         self.param_pool1 = nn.Linear(num_joints, 1)
        
#         #### class predict
#         self.dropout = nn.Dropout(p=0.1)
#         self.classifier = nn.Linear(self.proj_size, num_classes)

#         self.tanh = nn.Tanh()

#     def forward(self, x):
#         """
#         """
#         hs = []
#         for name in x:
#             if self.fpn_size is None:
#                 hs.append(getattr(self, "proj_"+name)(x[name]))
#             else:
#                 hs.append(x[name])
#         hs = torch.cat(hs, dim=1).transpose(1, 2).contiguous() # B, S', C --> B, C, S
#         hs = self.param_pool0(hs)
#         ### adaptive adjacency
#         q1 = self.conv_q1(hs).mean(1)
#         k1 = self.conv_k1(hs).mean(1)
#         A1 = self.tanh(q1.unsqueeze(-1) - k1.unsqueeze(1))
#         A1 = self.adj1 + A1 * self.alpha1
#         ### graph convolution
#         hs = self.conv1(hs)
#         hs = torch.matmul(hs, A1)
#         hs = self.batch_norm1(hs)
#         # ds = hs
#         ### predict
#         hs = self.param_pool1(hs)
#         ds = hs
#         hs = self.dropout(hs)
#         hs = hs.flatten(1)
#         hs = self.classifier(hs)

#         return hs ,ds

class WeaklySelector(nn.Module):

    def __init__(self, inputs: dict, num_classes: int, num_select: dict, fpn_size: Union[int, None] = None):
        """
        inputs: dictionary contain torch.Tensors, which comes from backbone
                [Tensor1(hidden feature1), Tensor2(hidden feature2)...]
                Please note that if len(features.size) equal to 3, the order of dimension must be [B,S,C],
                S mean the spatial domain, and if len(features.size) equal to 4, the order must be [B,C,H,W]

        """
        super(WeaklySelector, self).__init__()

        self.num_select = num_select

        self.fpn_size = fpn_size
        ### build classifier
        if self.fpn_size is None:
            self.num_classes = num_classes
            for name in inputs:
                fs_size = inputs[name].size()
                if len(fs_size) == 3:
                    in_size = fs_size[2]
                elif len(fs_size) == 4:
                    in_size = fs_size[1]
                m = nn.Linear(in_size, num_classes)
                self.add_module("classifier_l_"+name, m)

    # def select(self, logits, l_name):
    #     """
    #     logits: [B, S, num_classes]
    #     """
    #     probs = torch.softmax(logits, dim=-1)
    #     scores, _ = torch.max(probs, dim=-1)
    #     _, ids = torch.sort(scores, -1, descending=True)
    #     sn = self.num_select[l_name]
    #     s_ids = ids[:, :sn]
    #     not_s_ids = ids[:, sn:]
    #     return s_ids.unsqueeze(-1), not_s_ids.unsqueeze(-1)

    def forward(self, x, logits=None):
        """
        x : 
            dictionary contain the features maps which 
            come from your choosen layers.
            size must be [B, HxW, C] ([B, S, C]) or [B, C, H, W].
            [B,C,H,W] will be transpose to [B, HxW, C] automatically.
        """
        if self.fpn_size is None:
            logits = {}
        selections = {}
        for name in x:
            if len(x[name].size()) == 4:
                B, C, H, W = x[name].size()
                x[name] = x[name].view(B, C, H*W).permute(0, 2, 1).contiguous()
            C = x[name].size(-1)
            if self.fpn_size is None:
                logits[name] = getattr(self, "classifier_l_"+name)(x[name])
            
            probs = torch.softmax(logits[name], dim=-1)
            selections[name] = []
            preds_1 = []
            preds_0 = []
            num_select = self.num_select[name]
            for bi in range(logits[name].size(0)):
                max_ids, _ = torch.max(probs[bi], dim=-1)
                confs, ranks = torch.sort(max_ids, descending=True)
                sf = x[name][bi][ranks[:num_select]]
                nf = x[name][bi][ranks[num_select:]]  # calculate
                selections[name].append(sf) # [num_selected, C]
                preds_1.append(logits[name][bi][ranks[:num_select]])
                preds_0.append(logits[name][bi][ranks[num_select:]])
            
            selections[name] = torch.stack(selections[name])
            preds_1 = torch.stack(preds_1)
            preds_0 = torch.stack(preds_0)

            logits["select_"+name] = preds_1
            logits["drop_"+name] = preds_0

        return selections


class FPN(nn.Module):

    def __init__(self, inputs: dict, fpn_size: int, proj_type: str, upsample_type: str):
        """
        inputs : dictionary contains torch.Tensor
                 which comes from backbone output
        fpn_size: integer, fpn 
        proj_type: 
            in ["Conv", "Linear"]
        upsample_type:
            in ["Bilinear", "Conv", "Fc"]
            for convolution neural network (e.g. ResNet, EfficientNet), recommand 'Bilinear'. 
            for Vit, "Fc". and Swin-T, "Conv"
        """
        super(FPN, self).__init__()
        assert proj_type in ["Conv", "Linear"], \
            "FPN projection type {} were not support yet, please choose type 'Conv' or 'Linear'".format(proj_type)
        assert upsample_type in ["Bilinear", "Conv"], \
            "FPN upsample type {} were not support yet, please choose type 'Bilinear' or 'Conv'".format(proj_type)

        self.fpn_size = fpn_size
        self.upsample_type = upsample_type
        inp_names = [name for name in inputs]

        for i, node_name in enumerate(inputs):
            ### projection module
            if proj_type == "Conv":
                m = nn.Sequential(
                    nn.Conv2d(inputs[node_name].size(1), inputs[node_name].size(1), 1),
                    nn.ReLU(),
                    nn.Conv2d(inputs[node_name].size(1), fpn_size, 1)
                )
            elif proj_type == "Linear":
                m = nn.Sequential(
                    nn.Linear(inputs[node_name].size(-1), inputs[node_name].size(-1)),
                    nn.ReLU(),
                    nn.Linear(inputs[node_name].size(-1), fpn_size),
                )
            self.add_module("Proj_"+node_name, m)

            ### upsample module
            if upsample_type == "Conv" and i != 0:
                assert len(inputs[node_name].size()) == 3 # B, S, C
                in_dim = inputs[node_name].size(1)
                out_dim = inputs[inp_names[i-1]].size(1)
                if in_dim != out_dim:
                    m = nn.Conv1d(in_dim, out_dim, 1) # for spatial domain
                else:
                    m = nn.Identity()
                self.add_module("Up_"+node_name, m)

        if upsample_type == "Bilinear":
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        # 定义可学习的权重参数
        self.weight0 = nn.Parameter(torch.tensor(1.5))  # 浅层权重
        self.weight1 = nn.Parameter(torch.tensor(0.5))  # 深层权重
        # self.weight0 = nn.Parameter(torch.tensor(1.0))  # 浅层权重
        # self.weight1 = nn.Parameter(torch.tensor(1.0))  # 深层权重

    def upsample_add(self, x0: torch.Tensor, x1: torch.Tensor, x1_name: str):
        """
        return Upsample(x1) * weight1 + x0 * weight0
        """
        if self.upsample_type == "Bilinear":
            if x1.size(-1) != x0.size(-1):
                x1 = self.upsample(x1)
        else:
            x1 = getattr(self, "Up_"+x1_name)(x1)
        
        return x1 * self.weight1 + x0 * self.weight0

    def forward(self, x):
        """
        x : dictionary
            {
                "node_name1": feature1,
                "node_name2": feature2, ...
            }
        """
        ### project to same dimension
        hs = []
        for i, name in enumerate(x):
            x[name] = getattr(self, "Proj_"+name)(x[name])
            hs.append(name)

        for i in range(len(hs)-1, 0, -1):
            x1_name = hs[i]
            x0_name = hs[i-1]
            x[x0_name] = self.upsample_add(x[x0_name], 
                                           x[x1_name], 
                                           x1_name)
        return x


class PluginMoodel(nn.Module):

    def __init__(self, 
                 backbone: torch.nn.Module,
                 return_nodes: Union[dict, None],
                 img_size: int,
                 use_fpn: bool,
                 fpn_size: Union[int, None],
                 proj_type: str,
                 upsample_type: str,
                 use_selection: bool,
                 num_classes: int,
                 num_selects: dict, 
                 use_combiner: bool,
                 comb_proj_size: Union[int, None]
                 ):
        """
        * backbone: 
            torch.nn.Module class (recommand pretrained on ImageNet or IG-3.5B-17k(provided by FAIR))
        * return_nodes:
            e.g.
            return_nodes = {
                # node_name: user-specified key for output dict
                'layer1.2.relu_2': 'layer1',
                'layer2.3.relu_2': 'layer2',
                'layer3.5.relu_2': 'layer3',
                'layer4.2.relu_2': 'layer4',
            } # you can see the example on https://pytorch.org/vision/main/feature_extraction.html
            !!! if using 'Swin-Transformer', please set return_nodes to None
            !!! and please set use_fpn to True
        * feat_sizes: 
            tuple or list contain features map size of each layers. 
            ((C, H, W)). e.g. ((1024, 14, 14), (2048, 7, 7))
        * use_fpn: 
            boolean, use features pyramid network or not
        * fpn_size: 
            integer, features pyramid network projection dimension
        * num_selects:
            num_selects = {
                # match user-specified in return_nodes
                "layer1": 2048,
                "layer2": 512,
                "layer3": 128,
                "layer4": 32,
            }

        Note: after selector module (WeaklySelector) , the feature map's size is [B, S', C] which 
        contained by 'logits' or 'selections' dictionary (S' is selection number, different layer 
        could be different).
        """
        super(PluginMoodel, self).__init__()
        
        ### = = = = = Backbone = = = = =
        self.return_nodes = return_nodes
        if return_nodes is not None:
            self.backbone = create_feature_extractor(backbone, return_nodes=return_nodes)
        else:
            self.backbone = backbone
        
        ### get hidden feartues size
        rand_in = torch.randn(1, 3, img_size, img_size)
        outs = self.backbone(rand_in)

        ### just original backbone
        if not use_fpn and (not use_selection and not use_combiner):
            for name in outs:
                fs_size = outs[name].size()
                if len(fs_size) == 3:
                    out_size = fs_size[-1]
                elif len(fs_size) == 4:
                    out_size = fs_size[1]
                else:
                    raise ValusError("The size of output dimension of previous must be 3 or 4.")
            self.classifier = nn.Linear(out_size, num_classes)

        ### = = = = = FPN = = = = =
        self.use_fpn = use_fpn
        if self.use_fpn:
            self.fpn = FPN(outs, fpn_size, proj_type, upsample_type)
            self.build_fpn_classifier(outs, fpn_size, num_classes)

        self.fpn_size = fpn_size

        ### = = = = = Selector = = = = =
        self.use_selection = use_selection
        if self.use_selection:
            w_fpn_size = self.fpn_size if self.use_fpn else None # if not using fpn, build classifier in weakly selector
            self.selector = WeaklySelector(outs, num_classes, num_selects, w_fpn_size)

        ### = = = = = Combiner = = = = =
        self.use_combiner = use_combiner
        if self.use_combiner:
            assert self.use_selection, "Please use selection module before combiner"
            if self.use_fpn:
                gcn_inputs, gcn_proj_size = None, None
            else:
                gcn_inputs, gcn_proj_size = outs, comb_proj_size # redundant, fix in future
            total_num_selects = sum([num_selects[name] for name in num_selects]) # sum
            self.combiner = GCNCombiner(total_num_selects, num_classes, gcn_inputs, gcn_proj_size, self.fpn_size)

    def build_fpn_classifier(self, inputs: dict, fpn_size: int, num_classes: int):
        """
        Teh results of our experiments show that linear classifier in this case may cause some problem.
        """
        for name in inputs:
            m = nn.Sequential(
                    nn.Conv1d(fpn_size, fpn_size, 1),
                    nn.BatchNorm1d(fpn_size),
                    nn.ReLU(),
                    nn.Conv1d(fpn_size, num_classes, 1)
                )
            self.add_module("fpn_classifier_"+name, m)

    def forward_backbone(self, x):
        return self.backbone(x)

    def fpn_predict(self, x: dict, logits: dict):
        """
        x: [B, C, H, W] or [B, S, C]
           [B, C, H, W] --> [B, H*W, C]
        """
        for name in x:
            ### predict on each features point
            if len(x[name].size()) == 4:
                B, C, H, W = x[name].size()
                logit = x[name].view(B, C, H*W)
            elif len(x[name].size()) == 3:
                logit = x[name].transpose(1, 2).contiguous()
            logits[name] = getattr(self, "fpn_classifier_"+name)(logit)
            logits[name] = logits[name].transpose(1, 2).contiguous() # transpose

    def forward(self, x: torch.Tensor):

        logits = {}

        x = self.forward_backbone(x)

        if self.use_fpn:
            x = self.fpn(x)
            self.fpn_predict(x, logits)

        if self.use_selection:
            selects = self.selector(x, logits)

        if self.use_combiner:
            comb_outs ,feature_outs = self.combiner(selects)
            logits['comb_outs'] = comb_outs
            return logits, feature_outs
        
        if self.use_selection or self.use_fpn:
            return logits

        ### original backbone (only predict final selected layer)
        for name in x:
            hs = x[name]

        if len(hs.size()) == 4:
            hs = F.adaptive_avg_pool2d(hs, (1, 1))
            hs = hs.flatten(1)
        else:
            hs = hs.mean(1)
        out = self.classifier(hs)
        logits['ori_out'] = out

        return logits
