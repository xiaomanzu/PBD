import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F

class PiCO(nn.Module):

    def __init__(self, args, base_encoder):
        super().__init__()
        
        pretrained = (args.dataset == 'VLCS' or args.dataset == 'PACS')
        # we allow pretraining for VLCS and PACS, or the network will not converge
        print('pretrained=',pretrained)
        self.encoder_q = base_encoder(num_class=args.num_class,domain_class=args.domain_class, feat_dim=args.low_dim, pretrained=pretrained)# name=args.arch,
        # momentum encoder
        self.encoder_k = base_encoder(num_class=args.num_class,domain_class=args.domain_class, feat_dim=args.low_dim, pretrained=pretrained)#name=args.arch,

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(args.moco_queue, args.low_dim))
        self.register_buffer("queue_pseudo", torch.randn(args.moco_queue, args.num_class))
        self.register_buffer("queue_partial", torch.randn(args.moco_queue, args.num_class))
        self.register_buffer("queue_domain",torch.randn(args.moco_queue, args.domain_class))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))        
        self.register_buffer("prototypes", torch.zeros(args.num_class,args.low_dim))

    @torch.no_grad()
    def _momentum_update_key_encoder(self, args):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, partial_Y,domain_Y, args):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)
        partial_Y = concat_all_gather(partial_Y)
        domain_Y = concat_all_gather(domain_Y)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert args.moco_queue % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_pseudo[ptr:ptr + batch_size, :] = labels
        self.queue_partial[ptr:ptr + batch_size, :] = partial_Y
        self.queue_domain[ptr:ptr+batch_size,:] = domain_Y
        ptr = (ptr + batch_size) % args.moco_queue  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x, y, p_y,p_d):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        y_gather = concat_all_gather(y)
        p_y_gather = concat_all_gather(p_y)
        p_d_gather = concat_all_gather(p_d)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], y_gather[idx_this], p_y_gather[idx_this], p_d_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, y, p_y,p_d, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        y_gather = concat_all_gather(y)
        p_y_gather = concat_all_gather(p_y)
        p_d_gather = concat_all_gather(p_d)

        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], y_gather[idx_this], p_y_gather[idx_this], p_d_gather[idx_this]

    def reset_prototypes(self, prototypes):
        self.prototypes = prototypes

    def forward(self, img_q, im_k=None, partial_Y=None, args=None, eval_only=False):
        
        output, q,q_d = self.encoder_q(img_q)
        if eval_only:
            return output
        # for testing
        predicetd_scores = torch.softmax(output, dim=1) * partial_Y
        max_scores, pseudo_labels = torch.max(predicetd_scores, dim=1)
        # using partial labels to filter out negative labels

        # compute protoypical logits
        prototypes = self.prototypes.clone().detach()
        logits_prot = torch.mm(q, prototypes.t())
        score_prot = torch.softmax(logits_prot, dim=1)
        score_domain = torch.softmax(q_d,dim=1)

        # update momentum prototypes with pseudo labels
        for feat, label in zip(concat_all_gather(q), concat_all_gather(pseudo_labels)):
            self.prototypes[label] = self.prototypes[label]*args.proto_m + (1-args.proto_m)*feat
        # normalize prototypes    
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
        
        # compute key features 
        with torch.no_grad():  # no gradient 
            self._momentum_update_key_encoder(args)  # update the momentum encoder
            # shuffle for making use of BN
            im_k, predicetd_scores, partial_Y,score_domain, idx_unshuffle = self._batch_shuffle_ddp(im_k, predicetd_scores, partial_Y, score_domain)
            _, k, k_d= self.encoder_k(im_k)
            # undo shuffle
            k, predicetd_scores, partial_Y,score_domain = self._batch_unshuffle_ddp(k, predicetd_scores, partial_Y, score_domain, idx_unshuffle)

        features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        pseudo_scores = torch.cat((predicetd_scores, predicetd_scores, self.queue_pseudo.clone().detach()), dim=0)
        partial_target = torch.cat((partial_Y, partial_Y, self.queue_partial.clone().detach()), dim=0)
        score_domain_target =  torch.cat((score_domain, score_domain, self.queue_domain.clone().detach()), dim=0)
        # to calculate SupCon Loss using pseudo_labels and partial target
        
        # dequeue and enqueue
        self._dequeue_and_enqueue(k, predicetd_scores, partial_Y,score_domain, args)

        return output, features, pseudo_scores, partial_target,score_domain_target,q_d, score_prot

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

