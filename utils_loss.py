import torch
import torch.nn.functional as F
import torch.nn as nn

class partial_loss(nn.Module):
    def __init__(self, confidence, conf_ema_m=0.99):
        super().__init__()
        self.confidence = confidence
        self.init_conf = confidence.detach()
        self.conf_ema_m = conf_ema_m

    def set_conf_ema_m(self, epoch, args):
        start = args.conf_ema_range[0]
        end = args.conf_ema_range[1]
        self.conf_ema_m = 1. * epoch / args.epochs * (end - start) + start

    def forward(self, outputs, index):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * self.confidence[index, :]
        average_loss = - ((final_outputs).sum(dim=1)).mean()
        return average_loss
    
    def confidence_update(self, temp_un_conf, batch_index, batchY):
        with torch.no_grad():
            _, prot_pred = (temp_un_conf * batchY).max(dim=1)
            pseudo_label = F.one_hot(prot_pred, batchY.shape[1]).float().cuda().detach()
            self.confidence[batch_index, :] = self.conf_ema_m * self.confidence[batch_index, :]\
                 + (1 - self.conf_ema_m) * pseudo_label
        return None

class SupConLoss(nn.Module):
    """Following Supervised Contrastive Learning: 
        https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self,temperature=0.07, base_temperature=0.07,domain_number=7):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.domain_number = domain_number
        print('self.domain_number=',self.domain_number)

    def forward(self, features, mask=None, domain_score_cont=None,domain_target_cont=None, batch_size=-1):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if mask is not None and domain_score_cont is not None:
            # print('~~~~~~!!!!!',domain_score_cont.shape)
            domain_score_cont=domain_score_cont.detach()
            domain_target_cont=domain_target_cont.detach()
            # SupCon loss (Partial Label Mode)
            mask = mask.float().detach().to(device)
            anchor_dot_contrast = torch.div(
                torch.matmul(features[:batch_size], features.T),
                self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()  # (positive_number,positive_number+negative_number)

            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size).view(-1, 1).to(device),
                0
            )
            mask = mask * logits_mask
            
            exp_logits = torch.exp(logits)
            new_log_prob = torch.zeros(exp_logits.shape[0]).to(exp_logits.device)
            w_matrix = domain_score_cont[:batch_size]
            for i in range(self.domain_number):
                domain_mask = (domain_target_cont == i).view(-1)
                ne_po_logits = exp_logits[:,domain_mask].sum(1,keepdim=True)
                tranpose_domain_mask = (domain_target_cont[:batch_size] == i).expand(batch_size,domain_mask.shape[0])
                tmp_log_prob = torch.where(tranpose_domain_mask,logits - torch.log(ne_po_logits+1e-8),logits - torch.log(ne_po_logits + exp_logits+1e-12))
                w_d = w_matrix[:,i].reshape(-1,1)
                tmp_log_prob = tmp_log_prob * w_d
                tmp_log_prob = (mask * tmp_log_prob).sum(1) / mask.sum(1)
                new_log_prob += tmp_log_prob
            mean_log_prob_pos = new_log_prob
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
        else:
            # MoCo loss (unsupervised)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            q = features[:batch_size]
            k = features[batch_size:batch_size*2]
            queue = features[batch_size*2:]
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,kc->nk', [q, queue])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.temperature

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = F.cross_entropy(logits, labels)

        return loss