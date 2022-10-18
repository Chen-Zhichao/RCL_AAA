import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def EarlyAnnoScore(batch_centroids_tgt,
                    global_batch_centroids_src,
                    tgt_label,
                    tgt_predict,
                    numparts_h,
                    numparts_w
                    ):
                    
    h, w = tgt_label.size()[-2], tgt_label.size()[-1]
    parts_h = int(h / numparts_h)
    parts_w = int(w / numparts_w)

    early_anno_score = torch.zeros(tgt_label.size()).cuda()
    early_anno_score[:,:,:] = 0

    for img_idx in batch_centroids_tgt:
        for region_index in batch_centroids_tgt[img_idx]:
            i, j = int(region_index.split('_')[0]), int(region_index.split('_')[1])
            
            if [i,j] == [range(numparts_h)[-1], range(numparts_w)[-1]]:
                rg_id = [i*parts_h, h-1, j*parts_w, w-1]
            else:
                rg_id = [i*parts_h, (i+1)*parts_h-1, j*parts_w, (j+1)*parts_w-1]
            for cls in batch_centroids_tgt[img_idx][region_index]:
                if global_batch_centroids_src[img_idx]['0_0'].__contains__(cls) and global_batch_centroids_src[1-img_idx]['0_0'].__contains__(cls):
                    src_cls_prototype = (global_batch_centroids_src[img_idx]['0_0'][cls] + global_batch_centroids_src[1-img_idx]['0_0'][cls]) / len(global_batch_centroids_src)
                elif global_batch_centroids_src[img_idx]['0_0'].__contains__(cls):
                    src_cls_prototype = global_batch_centroids_src[img_idx]['0_0'][cls]
                elif global_batch_centroids_src[1-img_idx]['0_0'].__contains__(cls):
                    src_cls_prototype = global_batch_centroids_src[1-img_idx]['0_0'][cls]
                else:
                    continue
                tgt_cls_prototype = batch_centroids_tgt[img_idx][region_index][cls]
                cross_cls_unsimilarity = torch.tensor([1]).cuda() - F.cosine_similarity(src_cls_prototype, tgt_cls_prototype, dim=0)
                cls_mask = tgt_label[img_idx, rg_id[0]:rg_id[1], rg_id[2]:rg_id[3]].eq(cls)
                intra_cls_similarity = F.cosine_similarity(tgt_predict[img_idx, :, rg_id[0]:rg_id[1], rg_id[2]:rg_id[3]], tgt_cls_prototype.unsqueeze(dim=1).unsqueeze(dim=2), dim=0)
                score_mask = cls_mask * cross_cls_unsimilarity * intra_cls_similarity
                early_anno_score[img_idx, rg_id[0]:rg_id[1], rg_id[2]:rg_id[3]] += score_mask

    return early_anno_score