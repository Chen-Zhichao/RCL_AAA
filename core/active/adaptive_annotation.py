import math
import torch

import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from PIL import Image
from tqdm import tqdm
from core.active.early_anno_score import EarlyAnnoScore
from core.models.region_spliter import RegionSplit_CentroidCal



def AdaptiveAnnotation(cfg,
                        feature_extractor,
                        classifier,
                        src_train_loader,
                        tgt_epoch_loader,
                        now_iteration):

    feature_extractor.eval()
    classifier.eval()

    active_ratio = cfg.ACTIVE.RATIO / len(cfg.ACTIVE.SELECT_ITER)
    max_iter = cfg.SOLVER.MAX_ITER


    cross_entropy_computation = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    with torch.no_grad():
        for (src_data, tgt_data) in tqdm(zip(src_train_loader, tgt_epoch_loader)):
            src_input, src_label = src_data['img'], src_data['label']
            src_input = src_input.cuda(non_blocking=True)
            src_label = src_label.cuda(non_blocking=True)

            src_size = src_input.shape[-2:]
            src_out = classifier(feature_extractor(src_input), size=src_size)
            src_predict = torch.softmax(src_out, dim=1)

            tgt_input, path2mask = tgt_data['img'], tgt_data['path_to_mask']    # tgt_input: torch.Size([1, 3, 640, 1280])
            origin_mask, origin_label = tgt_data['origin_mask'], tgt_data['origin_label']       # origin_mask, origin_label: torch.Size([1, 1024, 2048])
            origin_size = tgt_data['size']
            active_indicator = tgt_data['active']
            selected_indicator = tgt_data['selected']
            path2indicator = tgt_data['path_to_indicator']

            tgt_input = tgt_input.cuda(non_blocking=True)

            tgt_size = tgt_input.shape[-2:]
            tgt_feat = feature_extractor(tgt_input)
            tgt_out = classifier(tgt_feat, size=tgt_size)       # tgt_out: torch.Size([1, 19, 640, 1280])
            tgt_predict = torch.softmax(tgt_out, dim=1)
            tgt_label = torch.argmax(tgt_predict[:,:,:,:],dim=1)    # tgt_label: torch.Size([1, 640, 1280])

            for i in range(len(origin_mask)):       # 一个batch内的图片数量  # origin_mask, origin_label: torch.Size([1, 1024, 2048])
                active_mask = origin_mask[i].cuda(non_blocking=True)
                ground_truth = origin_label[i].cuda(non_blocking=True)
                size = (origin_size[i][0], origin_size[i][1])       # size: tensor(1024), tensor(2048)
                num_pixel_cur = size[0] * size[1]
                active = active_indicator[i]        # torch.Size([1024, 2048])，最开始都是False
                selected = selected_indicator[i]

                output = tgt_out[i:i + 1, :, :, :]
                output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)


                tgt_label_interpolation = F.interpolate(tgt_label.float().unsqueeze(dim=0), size=size, mode='bilinear', align_corners=True)
                tgt_label_interpolation = tgt_label_interpolation.squeeze(dim=0).long()     # torch.Size([1, 1024, 2048]) 
                tgt_predict_interpolation = F.interpolate(tgt_predict, size=size, mode='bilinear', align_corners=True)  # torch.Size([1, 19, 1024, 2048])

                batch_centroids_tgt=RegionSplit_CentroidCal(predict=tgt_predict_interpolation, 
                                                            label=tgt_label_interpolation, 
                                                            is_source=False,
                                                            numparts_h=2, 
                                                            numparts_w=4,
                                                            tgt_centroids_base_ratio=0.9,
                                                            tgt_out=output)

                global_batch_centroids_src=RegionSplit_CentroidCal(predict=src_predict, 
                                                                        label=src_label, 
                                                                        is_source=True,
                                                                        numparts_h=1, 
                                                                        numparts_w=1)
        
                similarity_score = EarlyAnnoScore(batch_centroids_tgt=batch_centroids_tgt,
                                        global_batch_centroids_src=global_batch_centroids_src,
                                        tgt_label=tgt_label_interpolation,
                                        tgt_predict=tgt_predict_interpolation,
                                        numparts_h=2,
                                        numparts_w=4)

                # similarity_score: torch.Size([1, 1024, 2048])
                similarity_score = similarity_score.squeeze(dim=0) # similarity_score: torch.Size([1024, 2048])
                similarity_score[active] = 0.0     # 把上一轮已经标注过的pixel给置0

                # uncertainty_score: torch.Size([1, 1024, 2048])
                uncertainty_score = cross_entropy_computation(output, tgt_label_interpolation)
                uncertainty_score = uncertainty_score.squeeze(dim=0)

                active_budget = math.ceil(num_pixel_cur * active_ratio)    # 将要actively selected pixel的数量
                similarity_budget = math.ceil(active_budget * ((now_iteration - max_iter) / max_iter) ** 2)
                uncertainty_budget = active_budget - similarity_budget

                for pixel in range(active_budget):
                    if pixel < similarity_budget:      # similarity annotation
                        values, indices_h = torch.max(similarity_score, dim=0)
                    else:                                        # uncertainty annotation
                        values, indices_h = torch.max(uncertainty_score, dim=0)
                    _, indices_w = torch.max(values, dim=0)
                    w = indices_w.item()
                    h = indices_h[w].item()

                    # mask out
                    similarity_score[h,w] = 0.0
                    uncertainty_score[h,w] = 0.0
                    active[h,w] = True
                    selected[h,w] = True
                    # active sampling
                    active_mask[h,w] = ground_truth[h,w]

                active_mask = Image.fromarray(np.array(active_mask.cpu().numpy(), dtype=np.uint8))
                active_mask.save(path2mask[i])
                indicator = {
                    'active': active,
                    'selected': selected
                }
                torch.save(indicator, path2indicator[i])

    feature_extractor.train()
    classifier.train()