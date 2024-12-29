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
from load_datasets import get_dataset, HyperX, get_rask_dataset, get_test_dataset
#houston: 0.4  2    pavia: 0.4   6            DL: 0.5   1
parser = argparse.ArgumentParser(description='PyTorch ')
parser.add_argument('--test_pkl', type=str, default='/home/huangxizeng/code/HSItest/DTSM_test/results_param/PaviaU_to_C/0.001_kp0.4_act6_dim256_depth2_pt13_08-05_11-16-20/model_85.32.pkl',)
parser.add_argument('--gpu', type=int, default=1,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")
parser.add_argument('--patch_size', type=int, default=13,
                    help="Size of the spatial neighbourhood (optional, if ""absent will be set by the model)")
parser.add_argument('--batch_size', type=int, default=256,
                    help="Batch size (optional, if absent will be set by the model")


parser.add_argument('--flip_augmentation', action='store_true', default=False,
                    help="Random flips (if patch_size > 1)")
parser.add_argument('--radiation_augmentation', action='store_true',default=False,
                    help="Random radiation noise (illumination)")
parser.add_argument('--mixture_augmentation', action='store_true',default=False,
                    help="Random mixes between spectra")
args = parser.parse_args()

current_path = os.path.split(sys.argv[0])[0]# /home/huangxizeng/my_utils/正式版/HSItrain.py

rask = 'pavia' # 'houston | pavia | hyrank | sh'
parser = set_config(current_path, parser, rask)

args = parser.parse_args()

def experiment():#hyperparams是一个字典形式的配置变量
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    hyperparams = vars(args)#将args转化为dic的形式 {'a':1}
    root_path = os.path.dirname(os.path.dirname(args.test_pkl))
    print(f'root_path:{root_path}')
    rask_name = root_path.split('/')[-1]
    print(f'rask_name:{rask_name}')
    img_tar, gt_tar, LABEL_VALUES_tar, IGNORED_LABELS, RGB_BANDS, palette = get_test_dataset(rask_name)
    unpad_gt_tar = gt_tar
    # picshow_save(unpad_gt_src,'7',os.path.join(log_dir,f'src.tif'),None)
    # picshow_save(unpad_gt_tar,'7',os.path.join(log_dir,f'tar.tif'),None)     
    num_classes = gt_tar.max()
    channels = img_tar.shape[-1]#最后一个维度
    hyperparams.update({'n_classes': num_classes, 'n_bands': channels, 'ignored_labels': IGNORED_LABELS, 
                        'device': args.gpu, 'center_pixel': None, 'supervision': 'full'})
    r = int(hyperparams['patch_size']/2)+1
    img_tar=np.pad(img_tar,((r,r),(r,r),(0,0)),'symmetric')#填充
    gt_tar=np.pad(gt_tar,((r,r),(r,r)),'constant',constant_values=(0,0))#填充 
    print(f"填充完毕")
    hyperparams_train = hyperparams.copy()
    test_dataset = HyperX(img_tar, gt_tar,shuffle=False, **hyperparams)
    test_loader = data.DataLoader(test_dataset,
                                    pin_memory=True,
                                    batch_size=hyperparams['batch_size'])  
    len_tgt_loader = len(test_loader)
    len_tgt_dataset = len(test_loader.dataset)
    print(f"测试域训练样本loader加载完成, loader数量:{len_tgt_loader}, 样本dataset数量:{len_tgt_dataset}")                                          


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
    #测试
    acc,kappa,outputs=evaluate_tgt(model,test_loader,unpad_gt_tar,args.test_pkl,device)
    print(f"acc:{acc}")
    print(f"kappa:{kappa}")

    #出图
    pic_path=os.path.join(root_path, f'predict_{acc}.tif')
    picshow_save(outputs,'color_7', pic_path, None)    
    # ground_predict = HSI_utilis.picshow_save(outputs,hyperparams['target_name'],hyperparams['pic_save_path'],show=None)

if __name__=='__main__':
    experiment()

