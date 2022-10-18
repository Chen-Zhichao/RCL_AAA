import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        

    def forward(self, pos_set, neg_set, temperature):
        assert pos_set.size() != 0, "Positive pairs should not be EMPTY!"
        assert neg_set.size() != 0, "Negative pairs should not be EMPTY!"
        
        pos_head = torch.index_select(pos_set, 0, torch.tensor([0]).cuda(non_blocking=True))
        pos_pairs = torch.mm(pos_head, pos_set.permute(1,0))
        neg_pairs = torch.mm(pos_head, neg_set.permute(1,0))

        all_pairs = torch.cat([neg_pairs.repeat(pos_pairs.size()[1],1), pos_pairs.permute(1,0)], dim=1)
        all_pairs = torch.exp(all_pairs / temperature)

        exp_aggregation_row = all_pairs.sum(dim=1, keepdim=True)
        frac_row = torch.index_select(all_pairs, 1, torch.tensor([all_pairs.size()[1] - 1]).cuda(non_blocking=True)) / exp_aggregation_row
        log_row = torch.log(frac_row)
        
        if pos_set.size()[0] == 1:
            cl_loss = torch.mean(log_row) * (-1)
        else:
            cl_loss = torch.mean(log_row[1:,:]) * (-1)
    
        return cl_loss