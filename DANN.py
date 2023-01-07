import torch.nn as nn
import torch.nn.functional as F
import torch
import einops


class CFE(nn.Module):
    def __init__(self):
        super(CFE, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(310, 256),
            # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        x = self.module(x)
        return x

def pretrained_CFE(pretrained=False):
    model = CFE()
    if pretrained:
        pass
    return model



class DANN(nn.Module):
    def __init__(self, pretrained=False, number_of_category=4):
        super(DANN, self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        # self.cls_fc = nn.Linear(64, number_of_category)


    def forward(self, data):
        data=einops.rearrange(data,"b c w -> b (c w)")
        data = self.sharedNet(data)
        # x = self.cls_fc(data)
        return data

class SupConDANN(nn.Module):
    """backbone + projection head"""
    def __init__(self,  head='mlp', feat_dim=128, num_class=7, domain_class=3, pretrained=False):
        super(SupConDANN, self).__init__()
        model_fun, dim_in = DANN, 64
        self.encoder = model_fun()
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
        feat = self.encoder(x)#x[128,62,5]
        feat_c = self.head(feat)
        logits = self.fc(feat)
        feat_d = self.domain_head(feat.detach())
        return logits, F.normalize(feat_c, dim=1), F.normalize(feat_d,dim=1)