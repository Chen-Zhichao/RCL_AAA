import torch
import numpy as np
from .contrastive_loss import ContrastiveLoss


'''
To compute the contrastive loss at cross-domain level
'''

def CrossDomainContrastiveLoss(batch_centroids_src, 
                              batch_centroids_tgt,
                              positive_weight_increment_step,
                              negative_weight_increment_step,
                              temperature,
                              num_classes=19):

    loss = []

    contrastive_loss_criterion = ContrastiveLoss()

    for cl_round in [0,1]:
        if cl_round == 0:
            batch_centroids_0 = batch_centroids_src
            batch_centroids_1 = batch_centroids_tgt
        else:
            batch_centroids_1 = batch_centroids_src
            batch_centroids_0 = batch_centroids_tgt
            
        for k in batch_centroids_0:
            # calculate per image from another domain
            for j in batch_centroids_1:
                # calculate per class
                for cls in range(num_classes):      # pos: 另一张图片的positive samples，pos_origin：该张图片的positive samples
                    pos_origin, pos, neg = {}, {}, {}
                    # pos: {'region_{}_{}': {tensor([...])} }
                    # neg: {'region_{}_{}': {cls: tensor([...]), cls: tensor([...]), cls: tensor([...]), ...}}

                    for region in batch_centroids_1[j]:  # 找到每一个region里该cls对应的其它所有pos所在的region和neg所在的region
                        neg[region] = {}

                        for inter_cls in batch_centroids_1[j][region]:
                            neg[region][inter_cls] = batch_centroids_1[j][region][inter_cls]    # 只有这样copy tensor，才不至于改变batch_centroids本身

                        if batch_centroids_1[j][region].__contains__(cls):
                            pos[region] = batch_centroids_1[j][region][cls]     # 另一张图片的该区域的pos_region
                            del neg[region][cls]

                        if batch_centroids_0[k][region].__contains__(cls):        # 本张图片的该区域的pos_region
                            pos_origin[region] = batch_centroids_0[k][region][cls]

                    pos_origin_region, pos_region, neg_region = [], [], []
                    pos_region = list(pos.keys()) # positive pairs所在的batch内另一张图片的region
                    neg_region = list(neg.keys()) # negative pairs所在的batch内另一张图片region
                    pos_origin_region = list(pos_origin.keys()) # positive pairs所在的batch内此张图片的region

                    inter_region = list(set(pos_region + neg_region))   # 另一张图片上pos和neg sample的所在region
                    all_region = list(set(pos_origin_region + pos_region + neg_region)) # 合并多个list，并且删除重复元素
                    
                    for region_1 in all_region:
                        pos_per_region, neg_per_region = [], []         # 该类以该region为中心的positive和negative pairs
                        cl_per_region = []
                        region_1_index = np.array([int(region_1.split('_')[0]), int(region_1.split('_')[1])])
                        
                        if pos_origin.__contains__(region_1):
                            pos_per_region.append(pos_origin[region_1])    # positive pairs的头，即cls在该图中region_1的centroids     
                        
                        for neg_cls_1 in neg[region_1]:
                            neg_per_region.append(neg[region_1][neg_cls_1])   # negatiave pairs的头，即在region_1中除了cls之外的所有centroids

                        for region_2 in inter_region:                         # 收集另一张图片每个region的positive pairs和negative pairs
                            region_2_index = np.array([int(region_2.split('_')[0]), int(region_2.split('_')[1])])
                            positive_weight = 1 + positive_weight_increment_step * np.linalg.norm(region_1_index - region_2_index, ord=2) # L2 norm
                            negative_weight = 1 - negative_weight_increment_step * np.linalg.norm(region_1_index - region_2_index, ord=2) # L2 norm

                            if pos.__contains__(region_2):     # 其它区域的positive pairs，将会乘上权重
                                pos_per_region.append(pos[region_2] * positive_weight)

                            if neg.__contains__(region_2):     # 其它区域的negative pairs，将会乘上权重
                                for neg_cls_2 in neg[region_2]:
                                    neg_per_region.append(neg[region_2][neg_cls_2] * negative_weight)

                        if pos_per_region != [] and neg_per_region != []:    # 否则会报错，stack不能对empty list操作
                            pos_set_cl = torch.stack(pos_per_region, dim=0)       # 第一行tensor就是用于query的positive pair头，剩下都是所有乘上权重后的positive pairs
                            neg_set_cl = torch.stack(neg_per_region, dim=0)       # 所有乘上权重后的negative pairs
                            
                            cl_per_region = contrastive_loss_criterion(pos_set_cl, neg_set_cl, temperature)
                            
                            loss.append(cl_per_region)
                    
    if loss != []:
        loss = torch.stack(loss, dim=0).cuda(non_blocking=True)  
        loss = torch.mean(loss)    
    else:
        loss = torch.tensor([0]).cuda(non_blocking=True)

    return loss
