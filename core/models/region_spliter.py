import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter



def RegionSplit_CentroidCal(predict, 
                            label, 
                            numparts_h=4, 
                            numparts_w=8):
    if numparts_w / numparts_h != 2:
        raise ValueError("split part of an image should be w/h = 2.")
    h = label.size()[-2]
    w = label.size()[-1]
    batch_size = label.size()[0]
    parts_h = int(h / numparts_h)
    parts_w = int(w / numparts_w)
    batch_centroids = {}
    # batch_centroids:{
    #                  'img_idx': {
    #                               'region_{}_{}': {
    #                                                'centroids': {}
    for k in range(label.size()[0]):
        batch_centroids['img_idx_{}'.format(k)] = {}
        for i in range(numparts_h):
            for j in range(numparts_w):
                batch_centroids['img_idx_{}'.format(k)]['region_{}_{}'.format(i,j)] = {} 
                
                # Get region coordinates 
                if [i,j] == [range(numparts_h)[-1], range(numparts_w)[-1]]:
                    region_index = [i*parts_h, h-1, j*parts_w, w-1]
                else:
                    region_index = [i*parts_h, (i+1)*parts_h-1, j*parts_w, (j+1)*parts_w-1]
                # batch_centroids['img_idx_{}'.format(k)]['region_{}_{}'.format(i,j)]['region_index'] = region_index
                
                # Get all class ID in a single region
                classID = dict(Counter(label[k, region_index[0]:region_index[1], region_index[2]:region_index[3]].cpu().numpy().flatten()))
                if classID.__contains__(255): del classID[255]
                # batch_centroids['img_idx_{}'.format(k)]['region_{}_{}'.format(i,j)]['region_class']['classID'] = classID
                
                # Get all predict mean as centroids
                centroids = {}
                for key in classID:
                    predict_sum = torch.zeros([1,19], requires_grad=True)
                    predict_sum = predict_sum.cuda(non_blocking=True)
                    mask = (label[k, region_index[0]:region_index[1], region_index[2]:region_index[3]] == key)
                    predict_mask = predict[k, :, region_index[0]:region_index[1], region_index[2]:region_index[3]] * mask
                    centroids[key] = predict_mask.sum(axis=[1,2]) / classID[key]
                batch_centroids['img_idx_{}'.format(k)]['region_{}_{}'.format(i,j)]['centroids'] = centroids

    return batch_centroids


def IntraImageContrastiveLoss(batch_centroids, 
                              positive_weight_increment_step,
                              negative_weight_increment_step,
                              temperature,
                              numparts_h, 
                              numparts_w,
                              num_classes=19):
    intraimage_contrastiveloss = 0
    for k in batch_centroids:
        loss = []
        # calculate per clas
        for cls in range(num_classes):
            pos, neg = {}, {}
            # pos: {'region_{}_{}': {tensor([...])} }
            # neg: {'region_{}_{}': {cls: tensor([...]), cls: tensor([...]), cls: tensor([...]), ...}}
            for region in batch_centroids[k]:
                neg[region] = {}
                for intra_cls in batch_centroids[k][region]['centroids']:
                    neg[region][intra_cls] = batch_centroids[k][region]['centroids'][intra_cls]    # 只有这样copy tensor，才不至于改变batch_centroids本身
                if batch_centroids[k][region]['centroids'].__contains__(cls):
                    pos[region] = batch_centroids[k][region]['centroids'][cls]
                    del neg[region][cls]



    return intraimage_contrastiveloss


def contrastive_loss(pos_set, neg_set, temperature):
    assert pos_set.size() != 0, "Positive pairs should not be EMPTY!"
    assert neg_set.size() != 0, "Negative pairs should not be EMPTY!"

    pos_head = torch.index_select(pos_set, 0, torch.tensor([0]))
    pos_pairs = torch.mm(pos_head, pos_set.permute(1,0))
    neg_pairs = torch.mm(pos_head, neg_set.permute(1,0))

    all_pairs = torch.cat([neg_pairs.repeat(pos_pairs.size()[1],1), pos_pairs.permute(1,0)], dim=1)
    all_pairs = torch.exp(all_pairs / temperature)

    exp_aggregation_row = all_pairs.sum(dim=1, keepdim=True)
    frac_row = torch.index_select(all_pairs, 1, torch.tensor([all_pairs.size()[1] - 1])) / exp_aggregation_row
    log_row = torch.log(frac_row)
    
    if pos_set.size()[0] == 1:
        cl_loss = log_row * (-1)
    else:
        cl_loss = torch.mean(log_row[1:,:]) * (-1)
    
    return cl_loss