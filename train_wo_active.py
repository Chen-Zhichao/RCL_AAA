import argparse
import os
import datetime
import logging
import time

import torch
import torch.nn as nn
import torch.utils
import torch.distributed
from torch.utils.data import DataLoader
import multiprocessing

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_feature_extractor, build_classifier, RegionSplit_CentroidCal
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.active.build import PixelSelection, RegionSelection
from core.datasets.dataset_path_catalog import DatasetCatalog
from core.loss.negative_learning_loss import NegativeLearningLoss
from core.loss.local_consistent_loss import LocalConsistentLoss
from core.loss.intra_image_cl_loss import IntraImageContrastiveLoss
from core.loss.inter_image_cl_loss import InterImageContrastiveLoss
from core.loss.cross_domain_cl_loss import CrossDomainContrastiveLoss
from core.utils.utils import set_random_seed
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore')


def train(cfg):
    print("Here is {} CPU, {} GPU.".format(multiprocessing.cpu_count(), torch.cuda.device_count()))
    logger = logging.getLogger("RCL-AAA.trainer")
    tb_writer = SummaryWriter('./{}_{}_train_tensorboard_log'.format(cfg.DATASETS.SOURCE_TRAIN.split('_')[0], cfg.DATASETS.TARGET_TRAIN.split('_')[0]))
    print('Tensorboard writer log has been created at {}'.format('./{}_{}_train_tensorboard_log'.format(cfg.DATASETS.SOURCE_TRAIN.split('_')[0], cfg.DATASETS.TARGET_TRAIN.split('_')[0])))

    # create network
    device = torch.device(cfg.MODEL.DEVICE)
    feature_extractor = build_feature_extractor(cfg)
    #feature_extractor = nn.DataParallel(feature_extractor)
    feature_extractor.to(device)

    classifier = build_classifier(cfg)
    #classifier = nn.DataParallel(classifier)
    classifier.to(device)


    # init optimizer
    optimizer_fea = torch.optim.SGD(feature_extractor.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_fea.zero_grad()

    optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=cfg.SOLVER.BASE_LR * 10, momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_cls.zero_grad()
    
    iteration = 0
    
    # load checkpoint
    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.OUTPUT_DIR + '/' + cfg.resume, map_location=torch.device('cpu'))
        # iteration = checkpoint['iteration']       # because use warmup model, iteration begin from zero
        feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        optimizer_fea.load_state_dict(checkpoint['optimizer_fea'])
        classifier.load_state_dict(checkpoint['classifier'])
        optimizer_cls.load_state_dict(checkpoint['optimizer_cls'])
    # feature_extractor = nn.DataParallel(feature_extractor)      # modified by CZC
    # classifier = nn.DataParallel(classifier)            # modified by CZC
    # init mask for cityscape
    DatasetCatalog.initMask(cfg)

    # init data loader
    src_train_data = build_dataset(cfg, mode='train', is_source=True)
    tgt_train_data = build_dataset(cfg, mode='train', is_source=False)
    # tgt_epoch_data = build_dataset(cfg, mode='active', is_source=False, epochwise=True)

    src_train_loader = DataLoader(src_train_data, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=4,
                                  pin_memory=True, drop_last=True)
    tgt_train_loader = DataLoader(tgt_train_data, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=4,
                                  pin_memory=True, drop_last=True)
    # tgt_epoch_loader = DataLoader(tgt_epoch_data, batch_size=1, shuffle=False, num_workers=4,
                                #   pin_memory=True, drop_last=True) 
    
    # init loss
    sup_criterion = nn.CrossEntropyLoss(ignore_index=255)

    start_warmup_time = time.time()
    end = time.time()
    warmup_iters = cfg.SOLVER.WARMUP_ITER
    max_iters = cfg.SOLVER.MAX_ITER
    meters = MetricLogger(delimiter="  ")

    logger.info(">>>>>>>>>>>>>>>>   Start Training   >>>>>>>>>>>>>>>>")
    feature_extractor.train()
    classifier.train()
    active_round = 1
    for batch_index, (src_data, tgt_data) in enumerate(zip(src_train_loader, tgt_train_loader)):       
        data_time = time.time() - end

        current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration, max_iters,
                                          power=cfg.SOLVER.LR_POWER)
        # tb_writer.add_scalar(tag="lr", scalar_value=current_lr, global_step=iteration)      # added by czc
        for index in range(len(optimizer_fea.param_groups)):
            optimizer_fea.param_groups[index]['lr'] = current_lr
        for index in range(len(optimizer_cls.param_groups)):
            optimizer_cls.param_groups[index]['lr'] = current_lr * 10

        optimizer_fea.zero_grad()
        optimizer_cls.zero_grad()

        src_input, src_label = src_data['img'], src_data['label']
        src_input = src_input.cuda(non_blocking=True)
        src_label = src_label.cuda(non_blocking=True)

        # target data
        # tgt_mask is active label, 255 means unlabeled data
        tgt_input, tgt_mask = tgt_data['img'], tgt_data['mask']
        tgt_input = tgt_input.cuda(non_blocking=True)
        tgt_mask = tgt_mask.cuda(non_blocking=True)
        


        src_size = src_input.shape[-2:]
        src_out = classifier(feature_extractor(src_input), size=src_size)
        src_predict = torch.softmax(src_out, dim=1)

        tgt_size = tgt_input.shape[-2:]
        tgt_out = classifier(feature_extractor(tgt_input), size=tgt_size)
        tgt_predict = torch.softmax(tgt_out, dim=1)
        tgt_label = torch.argmax(tgt_predict[:,:,:,:],dim=1)

        batch_centroids_src = RegionSplit_CentroidCal(predict=src_predict, 
                                                        label=src_label, 
                                                        is_source=True,
                                                        numparts_h=cfg.LOSS.NUMPARTS_H, 
                                                        numparts_w=cfg.LOSS.NUMPARTS_W)
        batch_centroids_tgt = RegionSplit_CentroidCal(predict=tgt_predict, 
                                                        label=tgt_label, 
                                                        is_source=False,
                                                        numparts_h=cfg.LOSS.NUMPARTS_H, 
                                                        numparts_w=cfg.LOSS.NUMPARTS_W,
                                                        tgt_centroids_base_ratio=0.9,
                                                        tgt_out=tgt_out)
        global_batch_centroids_src = RegionSplit_CentroidCal(predict=src_predict, 
                                                            label=src_label, 
                                                            is_source=True,
                                                            numparts_h=1, 
                                                            numparts_w=1)

        # source supervision loss
        loss = torch.Tensor([0]).cuda()
        # loss_sup = sup_criterion(src_out, src_label)

        # intra-image level contrastive loss
        intra_cl_loss_src = IntraImageContrastiveLoss(batch_centroids=batch_centroids_src, 
                                                    positive_weight_increment_step=cfg.LOSS.POS_WEIGHT_INCRE_STEP,
                                                    negative_weight_increment_step=cfg.LOSS.NEG_WEIGHT_INCRE_STEP,
                                                    temperature=cfg.LOSS.TEMPERATURE,
                                                    num_classes=cfg.MODEL.NUM_CLASSES)                                                    
        tb_writer.add_scalar(tag="loss/intra_src", scalar_value=intra_cl_loss_src, global_step=iteration)
        meters.update(intra_cl_loss_src=intra_cl_loss_src.item())

        intra_cl_loss_tgt = IntraImageContrastiveLoss(batch_centroids=batch_centroids_tgt, 
                                                    positive_weight_increment_step=cfg.LOSS.POS_WEIGHT_INCRE_STEP,
                                                    negative_weight_increment_step=cfg.LOSS.NEG_WEIGHT_INCRE_STEP,
                                                    temperature=cfg.LOSS.TEMPERATURE,
                                                    num_classes=cfg.MODEL.NUM_CLASSES)                                                    
        tb_writer.add_scalar(tag="loss/intra_tgt", scalar_value=intra_cl_loss_tgt, global_step=iteration)
        meters.update(intra_cl_loss_tgt=intra_cl_loss_tgt.item())

        intra_cl_loss = intra_cl_loss_src + intra_cl_loss_tgt
        tb_writer.add_scalar(tag="loss/intra_cl_loss", scalar_value=intra_cl_loss, global_step=iteration)
        loss += intra_cl_loss

        # inter-image level contrastive loss
        inter_cl_loss_src = InterImageContrastiveLoss(batch_centroids=batch_centroids_src, 
                                                    positive_weight_increment_step=cfg.LOSS.POS_WEIGHT_INCRE_STEP,
                                                    negative_weight_increment_step=cfg.LOSS.NEG_WEIGHT_INCRE_STEP,
                                                    temperature=cfg.LOSS.TEMPERATURE,
                                                    num_classes=cfg.MODEL.NUM_CLASSES)                                                    
        tb_writer.add_scalar(tag="loss/inter_src", scalar_value=inter_cl_loss_src, global_step=iteration)
        meters.update(inter_cl_loss_src=inter_cl_loss_src.item())

        inter_cl_loss_tgt = InterImageContrastiveLoss(batch_centroids=batch_centroids_tgt, 
                                                    positive_weight_increment_step=cfg.LOSS.POS_WEIGHT_INCRE_STEP,
                                                    negative_weight_increment_step=cfg.LOSS.NEG_WEIGHT_INCRE_STEP,
                                                    temperature=cfg.LOSS.TEMPERATURE,
                                                    num_classes=cfg.MODEL.NUM_CLASSES)                                                    
        tb_writer.add_scalar(tag="loss/inter_tgt", scalar_value=inter_cl_loss_tgt, global_step=iteration)
        meters.update(inter_cl_loss_tgt=inter_cl_loss_tgt.item())

        inter_cl_loss = inter_cl_loss_src + inter_cl_loss_tgt
        tb_writer.add_scalar(tag="loss/inter_cl_loss", scalar_value=inter_cl_loss, global_step=iteration)
        
        loss += inter_cl_loss

        # cross-domain level contrastive loss
        cross_domain_cl_loss = CrossDomainContrastiveLoss(batch_centroids_src=batch_centroids_src, 
                                                        batch_centroids_tgt=batch_centroids_tgt,
                                                        positive_weight_increment_step=cfg.LOSS.POS_WEIGHT_INCRE_STEP,
                                                        negative_weight_increment_step=cfg.LOSS.NEG_WEIGHT_INCRE_STEP,
                                                        temperature=cfg.LOSS.TEMPERATURE,
                                                        num_classes=19)                                                    
        tb_writer.add_scalar(tag="loss/cross_domain", scalar_value=cross_domain_cl_loss, global_step=iteration)
        meters.update(cross_domain_cl_loss=cross_domain_cl_loss.item())

        loss += cross_domain_cl_loss



        # target active supervision loss
        if torch.sum((tgt_mask != 255)) != 0:  # target has labeled pixels
            loss_sup_tgt = sup_criterion(tgt_out, tgt_mask)
            meters.update(loss_sup_tgt=loss_sup_tgt.item())
            loss += loss_sup_tgt
            tb_writer.add_scalar(tag="loss/sup_tgt", scalar_value=loss_sup_tgt, global_step=iteration)

        loss.backward()
        optimizer_fea.step()
        optimizer_cls.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (cfg.SOLVER.MAX_ITER - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        iteration += 1

        if iteration % 1 == 0 or iteration == max_iters:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.02f} GB"
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer_fea.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0
                )
            )

        if iteration == cfg.SOLVER.MAX_ITER or iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            filename = os.path.join(cfg.OUTPUT_DIR, "model_iter{:06d}.pth".format(iteration))
            torch.save({'iteration': iteration,
                        'feature_extractor': feature_extractor.state_dict(),
                        'classifier': classifier.state_dict(),
                        'optimizer_fea': optimizer_fea.state_dict(),
                        'optimizer_cls': optimizer_cls.state_dict(),
                        }, filename)

        if iteration == cfg.SOLVER.MAX_ITER:
            break
        if iteration == cfg.SOLVER.STOP_ITER:
            break

    total_warmup_time = time.time() - start_warmup_time
    total_warmup_time_str = str(datetime.timedelta(seconds=total_warmup_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_warmup_time_str, total_warmup_time / cfg.SOLVER.STOP_ITER
        )
    )

    tb_writer.close()

def main():
    parser = argparse.ArgumentParser(description="Active Domain Adaptive Semantic Segmentation Training")
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str)
    parser.add_argument("--proctitle",
                        type=str,
                        default="RCL-AAA",
                        help="allow a process to change its title",)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()

    if args.opts is not None:
        args.opts[-1] = args.opts[-1].strip('\r\n')

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("RCL-AAA", output_dir, 0)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    logger.info('Initializing Cityscapes label mask...')

    set_random_seed(cfg.SEED)

    train(cfg)


if __name__ == '__main__':
    main()
