import torch
import torch.nn as nn
from collections import Counter
import math




def RegionSplit_CentroidCal(predict, 
                            label, 
                            is_source,
                            numparts_h=4, 
                            numparts_w=8,
                            tgt_centroids_base_ratio=1.0,     # 属于该类的所有样本中，uncertainty较低的一部分
                            tgt_out=None):
    # if numparts_w / numparts_h != 2:
    #     raise ValueError("split part of an image should be w/h = 2.")
    h, w = label.size()[-2], label.size()[-1]
    batch_size = label.size()[0]
    parts_h, parts_w = int(h / numparts_h), int(w / numparts_w)
    batch_centroids = {}
    # batch_centroids:{
    #                  'img_idx': {
    #                               'region_{}_{}': {
    #                                                'centroids': {}
    if is_source == True:
        src_label = label
    else:
        cross_entropy_computation = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        tgt_label = label
    for k in range(batch_size):
        batch_centroids[k] = {}
        for i in range(numparts_h):
            for j in range(numparts_w):
                batch_centroids[k]['{}_{}'.format(i,j)] = {} 
                
                # Get region coordinates 
                if [i,j] == [range(numparts_h)[-1], range(numparts_w)[-1]]:
                    rg_id = [i*parts_h, h-1, j*parts_w, w-1]                 # rg_id: region_index
                else:
                    rg_id = [i*parts_h, (i+1)*parts_h-1, j*parts_w, (j+1)*parts_w-1]
                # batch_centroids[k]['region_{}_{}'.format(i,j)]['rg_id'] = region_index
                
                # Get all class ID in a single region
                if is_source == True:
                    classID = dict(Counter(src_label[k, rg_id[0]:rg_id[1], rg_id[2]:rg_id[3]].cpu().numpy().flatten()))     # 放到代码文件中得改一下
                else:
                    classID = dict(Counter(tgt_label[k, rg_id[0]:rg_id[1], rg_id[2]:rg_id[3]].cpu().numpy().flatten()))     # 放到代码文件中得改一下
                
                if classID.__contains__(255): del classID[255]
                # batch_centroids[k]['region_{}_{}'.format(i,j)]['region_class']['classID'] = classID
                
                # Get all predict mean as centroids
                centroids = {}
                for key in classID:
                    # predict_sum = torch.zeros([1,19], requires_grad=True)
                    # predict_sum = predict_sum.cuda(non_blocking=True)
                    if is_source == True:
                        mask = src_label[k, rg_id[0]:rg_id[1], rg_id[2]:rg_id[3]].eq(key)
                    else:
                        origin_tgt_mask = tgt_label[k, rg_id[0]:rg_id[1], rg_id[2]:rg_id[3]].eq(key)       # 代表是这个类的

                        tgt_ce = cross_entropy_computation(tgt_out[k, :, rg_id[0]:rg_id[1], rg_id[2]:rg_id[3]].permute(1,0,2), \
                                                            tgt_label[k, rg_id[0]:rg_id[1], rg_id[2]:rg_id[3]])
                        tgt_cls_uncertainty = tgt_ce * origin_tgt_mask      # 取最小的几个值注意可能会取到0
                        
                        unselected_sample_num = math.ceil(origin_tgt_mask.sum().item() * (1 - tgt_centroids_base_ratio))
                        unselected_samples, _ = torch.topk(torch.flatten(tgt_cls_uncertainty), k=unselected_sample_num, dim=-1, largest=True)
                        uncertainty_thres = unselected_samples.min().item()
                        
                        uncertainty_mask = tgt_cls_uncertainty.le(uncertainty_thres)        # 代表uncertainty不会过高的样本

                        mask = origin_tgt_mask * uncertainty_mask
                        
                    predict_mask = predict[k, :, rg_id[0]:rg_id[1], rg_id[2]:rg_id[3]] * mask
                    centroids[key] = predict_mask.sum(axis=[1,2]) / classID[key]

                batch_centroids[k]['{}_{}'.format(i,j)] = centroids

    return batch_centroids