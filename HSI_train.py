import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as data
from tensorboardX import SummaryWriter
import os
import sys
import time
import yaml
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from tqdm import tqdm
from models import *
from HSI_utils import *
from load_datasets import get_dataset, HyperX, get_rask_dataset


parser = argparse.ArgumentParser(description='PyTorch ')
parser.add_argument('--rask_name', type=str, default='Houston13_to_18', help='Houston13_to_18 | PaviaU_to_C | S_to_H')
parser.add_argument('--gpu', type=int, default=2,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")
parser.add_argument('--patch_size', type=int, default=13,
                    help="Size of the spatial neighbourhood (optional, if ""absent will be set by the model)")
parser.add_argument('--batch_size', type=int, default=256,
                    help="Batch size (optional, if absent will be set by the model")
parser.add_argument('--lr', type=float, default=1e-3, 
                    help="Learning rate, set by the model if not specified.")
parser.add_argument('--seed', type=int, default=233,
                    help='random seed ')
parser.add_argument('--num_epoch', type=int, default=100,
                    help='the number of epoch')
parser.add_argument('--lr_scheduler', type=str, default='none', help='cosine | Exp |Step')
parser.add_argument('--test_stride', type=int, default=1,
                    help="Sliding window step stride during inference (default = 1)")
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--l2_decay', type=float, default=1e-4,
                    help='the L2  weight decay')
parser.add_argument('--training_sample_ratio', type=float, default=0.8,
                    help='training sample ratio')
parser.add_argument('--re_ratio', type=int, default=5,
                    help='multiple of of data augmentation')
parser.add_argument('--flip_augmentation', action='store_true', default=False,
                    help="Random flips (if patch_size > 1)")
parser.add_argument('--radiation_augmentation', action='store_true',default=False,
                    help="Random radiation noise (illumination)")
parser.add_argument('--mixture_augmentation', action='store_true',default=False,
                    help="Random mixes between spectra")

current_path = os.path.split(sys.argv[0])[0]# /home/huangxizeng/my_utils/正式版/HSItrain.py

rask = 'houston' # 'houston | pavia | hyrank | sh'
parser = set_config(current_path, parser, rask)

args = parser.parse_args()


def experiment():#hyperparams是一个字典形式的配置变量
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    hyperparams = vars(args)#将args转化为dic的形式 {'a':1}
    print(hyperparams)
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    root = os.path.join(args.save_path, args.rask_name)#返回哪个数据集迁移哪个数据集
    log_dir = os.path.join(root, str(args.lr)+'_kp'+str(args.patch_mixup_keep)+'_act'+str(args.act_num)+'_dim'+str(args.embed_dim)+'_depth'+str(args.depth)+
                           '_pt'+str(args.patch_size)+'_'+time_str)#模型的保存地址
    
    if not os.path.exists(root):#如果root不存在
        os.makedirs(root)#创建root目录
    if not os.path.exists(log_dir):#如果log_dir不存在
        os.makedirs(log_dir)#创建log_dir目录
    yaml_path = os.path.join(log_dir,'config.yaml')
    writer = SummaryWriter(log_dir)#首先实例化Writer
    log_path=os.path.join(log_dir,'log.txt')
    # log_path = os.path.join(log_dir, 'log.txt')
    log_file = open(log_path,'a+')


    seed_worker(args.seed) #生成随机种子 使结果可复现
    img_src, gt_src, img_tar, gt_tar, label_values_src, label_queue_tar, ignored_labels, RGB_BANDS, palette = get_rask_dataset(args.rask_name)
    unpad_gt_src = gt_src
    unpad_gt_tar = gt_tar
    # picshow_save(unpad_gt_src,'7',os.path.join(log_dir,f'src.tif'),None)
    # picshow_save(unpad_gt_tar,'7',os.path.join(log_dir,f'tar.tif'),None)     
    print(f'数据加载成功')
    sample_num_src = len(np.nonzero(gt_src)[0])#nonzero返回多个数组，有几维就返回几个数组
    sample_num_tar = len(np.nonzero(gt_tar)[0])
    tmp = args.training_sample_ratio*args.re_ratio*sample_num_src/sample_num_tar
    num_classes = int(gt_tar.max())
    channels = img_src.shape[-1]#最后一个维度
    hyperparams.update({'n_classes': num_classes, 'n_bands': channels, 'ignored_labels': ignored_labels, 
                         'center_pixel': None, 'supervision': 'full'})
    
    r = int(hyperparams['patch_size']/2)+1
    img_src=np.pad(img_src,((r,r),(r,r),(0,0)),'symmetric')#填充
    img_tar=np.pad(img_tar,((r,r),(r,r),(0,0)),'symmetric')#填充
    gt_src=np.pad(gt_src,((r,r),(r,r)),'constant',constant_values=(0,0))#填充
    gt_tar=np.pad(gt_tar,((r,r),(r,r)),'constant',constant_values=(0,0))#填充 
    print(f"填充完毕")
    test_gt_tar, _, _, _ = sample_gt(gt_tar, 1, mode='random')
    train_gt_src, _, _, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')
    # _, val_gt_src, _, _ = sample_gt(gt_tar, args.training_sample_ratio, mode='random')
    img_src_con, train_gt_src_con = img_src, train_gt_src
    img_tar_con, test_gt_tar_con = img_tar, test_gt_tar
    # val_gt_src_con = val_gt_src
    if tmp < 1:
        for i in range(args.re_ratio-1):
            img_src_con = np.concatenate((img_src_con,img_src))
            train_gt_src_con = np.concatenate((train_gt_src_con,train_gt_src))
            # val_gt_src_con = np.concatenate((val_gt_src_con,val_gt_src))

    hyperparams_train = hyperparams.copy()
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dataset = HyperX(img_src_con, train_gt_src_con,shuffle=True,**hyperparams_train)
    train_loader = data.DataLoader(train_dataset,
                                    batch_size=hyperparams['batch_size'],
                                    pin_memory=True,
                                    worker_init_fn=seed_worker,
                                    generator=g,
                                    shuffle=True,)
    len_src_loader = len(train_loader)
    len_src_dataset = len(train_loader.dataset)
    print(f"源域训练样本loader加载完成, loader数量:{len_src_loader}, 样本dataset数量:{len_src_dataset}")  

    # val_dataset = HyperX(img_tar_con, val_gt_src_con, shuffle=True,**hyperparams)
    # val_loader = data.DataLoader(val_dataset,
    #                                 pin_memory=True,
    #                                 batch_size=hyperparams['batch_size'])
    # len_val_loader = len(val_loader)
    # len_val_dataset = len(val_loader.dataset)
    # print(f"验证域训练样本loader加载完成, loader数量:{len_val_loader}, 样本dataset数量:{len_val_dataset}") 

    test_dataset = HyperX(img_tar_con, test_gt_tar_con,shuffle=False, **hyperparams)
    test_loader = data.DataLoader(test_dataset,
                                    pin_memory=True,
                                    worker_init_fn=seed_worker,
                                    generator=g,
                                    batch_size=hyperparams['batch_size'])  
    len_tgt_loader = len(test_loader)
    len_tgt_dataset = len(test_loader.dataset)
    print(f"测试域训练样本loader加载完成, loader数量:{len_tgt_loader}, 样本dataset数量:{len_tgt_dataset}")                                          
    save_dict_to_yaml(hyperparams, yaml_path)
    print('-------------------------Train-------------------------------')
    print(f'---------------------{args.rask_name}--------------------------')
    # 方法
    model = DTAM(img_size=args.patch_size,patch_size=args.sub_patch_size, depth=args.depth,
                      act_num=args.act_num,patch_mixup_keep=args.patch_mixup_keep,  
                      embed_dim=args.embed_dim, channels=channels, num_classes=num_classes,
                      use_middle_cls_token=args.use_middle_cls_token,
                      rms_norm=args.rms_norm, 
                      residual_in_fp32=args.residual_in_fp32, 
                      fused_add_norm=args.fused_add_norm, 
                      final_pool_type=args.final_pool_type, 
                      if_abs_pos_embed=args.if_abs_pos_embed, 
                      bimamba_type=args.bimamba_type,
                      if_cls_token=args.if_cls_token, device=device).to(device)
    print(model, file=log_file)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-3)
    criterion = nn.CrossEntropyLoss()
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch)
    elif args.lr_scheduler == 'Exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif args.lr_scheduler == 'Step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(args.num_epoch*0.8))
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(args.num_epoch*0.8))

    best_epoch = 0
    best_acc = 0
    for epoch in range(args.num_epoch):
        loss_list = []
        model.train()
        t1 = time.time()
        t = tqdm(train_loader)
        t.set_description("Epoch [{}/{}]".format(epoch,args.num_epoch))
        for step, (inputs, targets) in enumerate(t):
            # move train data to GPU
            inputs = inputs.to(device)
            targets = targets - 1
            targets = targets.to(device)
            out1, targets_b, lam = model.forward_train(inputs, targets, if_random_cls_token_position=args.if_random_cls_token_position, if_random_token_rank=args.if_random_token_rank)
            # out1 = model(inputs, if_random_cls_token_position=args.if_random_cls_token_position, if_random_token_rank=args.if_random_token_rank)
            loss = mixup_criterion(criterion, out1, targets, targets_b, lam)
            loss_list.append([loss.item()])

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

        cls_loss=np.mean(loss_list, 0)[0]
        t2 = time.time()
        model.eval()

        teacc, kappa, _ = evaluate(model, test_loader, unpad_gt_tar, device)
        if best_acc < teacc:
            best_acc = teacc
            best_epoch = epoch
            _, _, _ = evaluate(model, test_loader, unpad_gt_tar, device, tgt=True, file=log_file)
            if best_acc > 55:
                pklpath=os.path.join(log_dir, f'model_{best_acc:.2f}.pkl')
                torch.save(model.state_dict(), pklpath)

        print(f'epoch: {epoch}, time: {t2-t1:.2f}, cls_loss: {cls_loss:.4f} /// teacc: {teacc:2.2f} kappa: {kappa:4.4f} best_acc: {best_acc:.2f}  best_epoch: {best_epoch}')
        log_file.write(f'epoch: {epoch}, time: {t2-t1:.2f}, cls_loss: {cls_loss:.4f} /// teacc: {teacc:2.2f} kappa: {kappa:4.4f} best_acc: {best_acc:.2f}  best_epoch: {best_epoch} \n') 
        log_file.flush()
        scheduler.step()
    log_file.close()


if __name__=='__main__':

    experiment()

