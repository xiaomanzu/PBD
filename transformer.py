import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class Transformer(nn.Module):
    def __init__(self,n_model,n_head,n_layer):
        super(Transformer, self).__init__()
        self.encoder = \
            nn.ModuleList([nn.Linear(n_model,n_model) for _ in range(n_layer)])
    def forward(self,x):
        # batchsize, times, n_model=62*5
        # x = einops.rearrange(x,"b c w -> b (c w)")
        for i in self.encoder:
            x = i(x)
        return x

class SupConTransformer(nn.Module):
    """backbone + projection head"""
    def __init__(self, head='mlp', feat_dim=128, num_class=7, domain_class=3, pretrained=False):
        super(SupConTransformer, self).__init__()
        model_fun, dim_in = Transformer, 310
        self.encoder = model_fun(310,1,2)
        self.fc = nn.Linear(dim_in, num_class)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
            self.domain_head = nn.Linear(dim_in, domain_class)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
            self.domain_head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, domain_class)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        self.register_buffer("prototypes", torch.zeros(7, feat_dim))

    def forward(self, x):
        feat = self.encoder(x)
        feat_c = self.head(feat)
        logits = self.fc(feat)
        feat_d = self.domain_head(feat.detach())
        return logits, F.normalize(feat_c, dim=1), F.normalize(feat_d,dim=1)