import argparse
import os, sys
import random
import shutil
import time
import warnings
from enum import Enum
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchstat import stat
from torch.cuda.amp import autocast, GradScaler

lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

lib_path = os.path.abspath(os.path.join('..','mycode','HRNetCls','lib'))
sys.path.append(lib_path)

lib_path = os.path.abspath(os.path.join('..','mycode','TorchCamMain'))
sys.path.append(lib_path)


#print(sys.path)
import mycode
import mycode.wlg_dataset
from mycode.wlg_dataset import Wlg_Dataset

from mycode.HRNetCls.lib.models.wlg_hrnet_cascade import HighResolutionNet
#from mycode.HRNetCls.lib.models.wlg_hrnet_cascade_ca import HighResolutionNet
#from mycode.HRNetCls.lib.models.wlg_hrnet_cascade_cbam import HighResolutionNet
#from mycode.HRNetCls.lib.models.wlg_hrnet_cascade_cbam_Sh import HighResolutionNet
#from mycode.HRNetCls.lib.models.wlg_hrnet_cascade_se import HighResolutionNet


from mycode.HRNetCls.lib.config import config
from mycode.HRNetCls.lib.config import update_config
from yacs.config import CfgNode as CN
import cv2
from mycode.MethodComp import method
from mycode.TorchCamMain.torchcam.methods import GradCAM, LayerCAM,XGradCAM, GradCAMpp, SmoothGradCAMpp
from matplotlib import cm


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', default='imagenet',
                    help='path to dataset (default: imagenet)')

parser.add_argument('-a', '--arch', metavar='ARCH', default='hrnet', #hrnet  hrnet resnet18 #MobileNetV2 #AlexNet
                    #choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

parser.add_argument('--ADA', default=1, type=int,
                    help='use ada')
parser.add_argument('--RED', default=1, type=int,
                    help='use red')
parser.add_argument('--MFL', default=0, type=int,
                    help='use MixupFL')
parser.add_argument('--MUS', default=0, type=int,
                    help='use multi-Scale')
parser.add_argument('--FFS', default=0, type=int,
                    help='use flip fuse')
parser.add_argument('--RER', default=0, type=int,
                    help='use random erase')
parser.add_argument('--RLS', default=1, type=int,
                    help='Ruler lenth scale')
              
parser.add_argument('--img_w', default=64, type=int,
                    help='image width')
parser.add_argument('--img_h', default=640, type=int,
                    help='image height')
parser.add_argument('--CAR', default=0.5, type=float,
                    help='cascade Ratio')      #cascade 第二阶段比例  
parser.add_argument('--cas_img_w', default=64, type=int,
                    help='image width')
parser.add_argument('--cas_img_h', default=640, type=int,
                    help='image height')                   
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,  #for cls its 0.1 but too high to reg
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--ot', '--Out-times', default=100, type=float,  #for cls its 0.1 but too high to reg
                    metavar='OT', help='out put times for smoothL1', dest='ot')
parser.add_argument('--caot', '--cascade-Out-times', default=100, type=float,  #for cls its 0.1 but too high to reg
                    metavar='CAOT', help='casecade out put times for smoothL1', dest='caot')
parser.add_argument('--glalp', '--loss-alpha', default=1, type=float,  #for cls its 0.1 but too high to reg
                    metavar='GLALP', help='loss consistent', dest='glalp')
parser.add_argument('--use_amp', default=False, type=bool,
                    help='is use amp')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--markly', default='', type=str, metavar='markly',
                    help='storge prestring')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-tas', '--testandshow', dest='testandshow', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu',default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

def seed_torch(seed = 19891012):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def main():

    global bestValResult_list
    global onlyflag
    global training_info 
    global logerPath 

    args = parser.parse_args()
    onlyflag = time.strftime('%y_%m_%d_%H_%M_%S', time.localtime())
    training_info = '{}_CAR{:.2f}_w{}h{}Cw{}Ch{}_model{}_MUS{}_{}'.format(args.markly, args.CAR, args.img_w, args.img_h, args.cas_img_w, args.cas_img_h, args.arch, args.MUS, onlyflag) 
    logerPath = './loger_{}.txt'.format(training_info) 

    if args.seed is not None:
        '''
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        '''
        seed_torch(seed = args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):



    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    
    # Various additional model inputs have been added here, including the hrnet and hrnetsmall structures各种额外的模型输入，原有内置模型应用
    if  args.arch !='hrnet_w18sv2_1_w18sv2_1' and args.arch !='hrnet_w18sv2_1_w18sv2_2' and args.arch !='hrnet_w18sv2_1_w18sv2_3' and args.arch !='hrnet_w18sv2_1_w18sv2_4' and  \
             args.arch !='hrnet_w18sv2_2_w18sv2_1' and args.arch !='hrnet_w18sv2_2_w18sv2_2' and args.arch !='hrnet_w18sv2_2_w18sv2_3' and args.arch !='hrnet_w18sv2_2_w18sv2_4' and  \
             args.arch !='hrnet_w18sv2_3_w18sv2_1' and args.arch !='hrnet_w18sv2_3_w18sv2_2' and args.arch !='hrnet_w18sv2_3_w18sv2_3' and args.arch !='hrnet_w18sv2_3_w18sv2_4' and  \
             args.arch !='hrnet_w18sv2_4_w18sv2_1' and args.arch !='hrnet_w18sv2_4_w18sv2_2' and args.arch !='hrnet_w18sv2_4_w18sv2_3' and args.arch !='hrnet_w18sv2_4_w18sv2_4' :

        print(models.__dict__)
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True,num_classes =1)

        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]( num_classes = 1)
        '''
        model.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1))
        '''   
    # Various additional model inputs have been added here, including the hrnet and hrnetsmall structures各种额外的模型输入，这里增加了hrnet 和hrnetsmall结构      
    else:
        pretrained_path = ''
        if 'hrnet_w18sv2_2_w18sv2_1' == args.arch:
            hrnetcfgpath = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/cls_hrnet_w18sv2_2b_w18sv2_1b_cascade.yaml'            
            Shrnetcfgpath = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/cls_hrnet_w18sv2_4b_w18sv2_1b_cascade.yaml'
            pretrained_path = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/hrnet_w18_small_model_v2.pth'
            Spretrained_path = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/hrnet_w18_small_model_v2.pth'
        elif 'hrnet_w18sv2_2_w18sv2_2' == args.arch:
            hrnetcfgpath = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/cls_hrnet_w18sv2_2b_w18sv2_2b_cascade.yaml'            
            Shrnetcfgpath = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/cls_hrnet_w18sv2_4b_w18sv2_1b_cascade.yaml'
            pretrained_path = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/hrnet_w18_small_model_v2.pth'
            Spretrained_path = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/hrnet_w18_small_model_v2.pth'
        elif 'hrnet_w18sv2_2_w18sv2_3' == args.arch:
            hrnetcfgpath = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/cls_hrnet_w18sv2_2b_w18sv2_3b_cascade.yaml'            
            Shrnetcfgpath = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/cls_hrnet_w18sv2_4b_w18sv2_1b_cascade.yaml'
            pretrained_path = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/hrnet_w18_small_model_v2.pth'
            Spretrained_path = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/hrnet_w18_small_model_v2.pth'
        elif 'hrnet_w18sv2_2_w18sv2_4' == args.arch:
            hrnetcfgpath = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/cls_hrnet_w18sv2_2b_w18sv2_4b_cascade.yaml'            
            Shrnetcfgpath = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/cls_hrnet_w18sv2_4b_w18sv2_1b_cascade.yaml'
            pretrained_path = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/hrnet_w18_small_model_v2.pth'
            Spretrained_path = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/hrnet_w18_small_model_v2.pth'
        elif 'hrnet_w18sv2_3_w18sv2_1' == args.arch:
            hrnetcfgpath = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/cls_hrnet_w18sv2_3b_w18sv2_1b_cascade.yaml'            
            Shrnetcfgpath = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/cls_hrnet_w18sv2_4b_w18sv2_1b_cascade.yaml'
            pretrained_path = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/hrnet_w18_small_model_v2.pth'
            Spretrained_path = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/hrnet_w18_small_model_v2.pth'
        elif 'hrnet_w18sv2_3_w18sv2_2' == args.arch:
            hrnetcfgpath = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/cls_hrnet_w18sv2_3b_w18sv2_2b_cascade.yaml'            
            Shrnetcfgpath = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/cls_hrnet_w18sv2_4b_w18sv2_1b_cascade.yaml'
            pretrained_path = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/hrnet_w18_small_model_v2.pth'
            Spretrained_path = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/hrnet_w18_small_model_v2.pth'
        elif 'hrnet_w18sv2_3_w18sv2_3' == args.arch:
            hrnetcfgpath = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/cls_hrnet_w18sv2_3b_w18sv2_3b_cascade.yaml'            
            Shrnetcfgpath = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/cls_hrnet_w18sv2_4b_w18sv2_1b_cascade.yaml'
            pretrained_path = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/hrnet_w18_small_model_v2.pth'
            Spretrained_path = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/hrnet_w18_small_model_v2.pth'
        elif 'hrnet_w18sv2_3_w18sv2_4' == args.arch:
            hrnetcfgpath = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/cls_hrnet_w18sv2_3b_w18sv2_4b_cascade.yaml'            
            Shrnetcfgpath = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/cls_hrnet_w18sv2_4b_w18sv2_1b_cascade.yaml'
            pretrained_path = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/hrnet_w18_small_model_v2.pth'
            Spretrained_path = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/hrnet_w18_small_model_v2.pth'
        elif 'hrnet_w18sv2_4_w18sv2_1' == args.arch:
            hrnetcfgpath = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/cls_hrnet_w18sv2_4b_w18sv2_1b_cascade.yaml'            
            Shrnetcfgpath = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/cls_hrnet_w18sv2_4b_w18sv2_1b_cascade.yaml'
            pretrained_path = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/hrnet_w18_small_model_v2.pth'
            Spretrained_path = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/hrnet_w18_small_model_v2.pth'
        elif 'hrnet_w18sv2_4_w18sv2_2' == args.arch:
            hrnetcfgpath = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/cls_hrnet_w18sv2_4b_w18sv2_2b_cascade.yaml'            
            Shrnetcfgpath = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/cls_hrnet_w18sv2_4b_w18sv2_1b_cascade.yaml'
            pretrained_path = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/hrnet_w18_small_model_v2.pth'
            Spretrained_path = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/hrnet_w18_small_model_v2.pth'
        elif 'hrnet_w18sv2_4_w18sv2_3' == args.arch:
            hrnetcfgpath = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/cls_hrnet_w18sv2_4b_w18sv2_3b_cascade.yaml'            
            Shrnetcfgpath = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/cls_hrnet_w18sv2_4b_w18sv2_1b_cascade.yaml'
            pretrained_path = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/hrnet_w18_small_model_v2.pth'
            Spretrained_path = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/hrnet_w18_small_model_v2.pth'
        elif 'hrnet_w18sv2_4_w18sv2_4' == args.arch:
            hrnetcfgpath = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/cls_hrnet_w18sv2_4b_w18sv2_4b_cascade.yaml'            
            Shrnetcfgpath = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/cls_hrnet_w18sv2_4b_w18sv2_1b_cascade.yaml'
            pretrained_path = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/hrnet_w18_small_model_v2.pth'
            Spretrained_path = '/data1/private/Project_branch/3.wlg3.0/wlg_reg/mycode/HRNetCls/configs_y/hrnet_w18_small_model_v2.pth'


        else:
            print('please check model name!!!')

        #cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
        #cls_hrnet_w18_small_v1_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
        #cls_hrnet_w18_small_v2_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
        #cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
        
        #HRNet_W18_C_ssld_pretrained.pth
        #hrnet_w18_small_model_v1.pth
        #hrnet_w18_small_model_v2.pth
        #HRNet_W48_C_ssld_pretrained.pth
        cfg_f = open(hrnetcfgpath)
        cfg = CN.load_cfg(cfg_f)
        cfg.freeze()
        cfg['MODEL']['IMAGE_SIZE'] = [args.img_w, args.img_h]
        cfg['MODEL']['CASCADE_IMAGE_SIZE'] = [args.cas_img_w, args.cas_img_h]
        cfg['MODEL']['CASCADE_RATIO'] = args.CAR
        print(cfg['MODEL']['CASCADE_RATIO'])
        model = HighResolutionNet(cfg)


        Scfg_f = open(Shrnetcfgpath)
        Scfg = CN.load_cfg(Scfg_f)
        Scfg.freeze()   
        Scfg['MODEL']['IMAGE_SIZE'] = [args.cas_img_w, args.cas_img_h]
        Scfg['MODEL']['CASCADE_IMAGE_SIZE'] = [args.img_w, args.img_h]
        Scfg['MODEL']['CASCADE_RATIO'] = args.CAR   
        Smodel = HighResolutionNet(Scfg)


        pretrained_model = torch.load(pretrained_path, map_location = {'cuda:7': 'cuda:0'})
        Spretrained_model = torch.load(Spretrained_path, map_location = {'cuda:7': 'cuda:0'})
        model.load_state_dict(pretrained_model, strict = False)
        Smodel.load_state_dict(Spretrained_model, strict = False)
        print("=> loaded pretrained_model '{}'".format(pretrained_path))
        print("=> loaded Spretrained_model '{}'".format(Spretrained_path))

    print(model)
    #stat(model, (3, args.img_h, args.img_w))



    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            Smodel.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            Smodel = torch.nn.parallel.DistributedDataParallel(Smodel, device_ids=[args.gpu])
        else:
            model.cuda()
            Smodel.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            Smodel = torch.nn.parallel.DistributedDataParallel(Smodel)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        Smodel = Smodel.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            Smodel.features = torch.nn.DataParallel(Smodel.features)
            model.cuda()
            Smodel.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            Smodel = torch.nn.DataParallel(Smodel).cuda()





    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            bestValResult_list = checkpoint['bestValResult_list']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                bestValResult_list = bestValResult_list.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    #traindir = os.path.join(args.data, '0.try_train/*.json')
    #valdir = os.path.join(args.data, '0.try_val/*.json')
 


    valpath_list = ['val/*.json', ]
    valdir_list = [os.path.join(args.data, valpath) for valpath in valpath_list]


    valAve_list = [0.6368139752585013, ]   
    valRelAve_list = [1.3551886202383459, ]  


    print(args.data)

    valdataset_list = [Wlg_Dataset(valdir, img_size = [64, 6400]) for valdir in valdir_list]

    valloader_list = [torch.utils.data.DataLoader(
        valdataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True) for valdataset in valdataset_list]    


    # test and verify验证
	###??

    if args.evaluate:
        valEvaResult_list = []
        for pidx in range(len(valpath_list)):
            valEvaResult = validate(valloader_list[pidx], model, args, in_ave_gt = valAve_list[pidx], in_rel_ave_gt = valRelAve_list[pidx])
            valEvaResult_list.append(valEvaResult)
            print('EVA_PATH: {}'.format(valpath_list[pidx]))
            print('acc:{:0.3f}_{:0.3f}_{:0.3f}_{:0.3f}'.format(valEvaResult[0],valEvaResult[1],valEvaResult[2],valEvaResult[3]))
            print('RMAE:{:0.4f}, RMSE:{:0.4f}, R^2:{:0.4f}'.format(valEvaResult[4],valEvaResult[5],valEvaResult[6]) )
            print('rel, acc:{:0.3f}_{:0.3f}_{:0.3f}_{:0.3f}'.format(valEvaResult[7],valEvaResult[8],valEvaResult[9],valEvaResult[10]))           
            print('rel, RMAE:{:0.4f}, RMSE:{:0.4f}, R^2:{:0.4f}'.format(valEvaResult[11],valEvaResult[12],valEvaResult[13]) )
        
        return


def set_bn_eval(m):
    classname = m.__class__.__name__
    #pring('set_bn_eval')
    if classname.find('BatchNorm')!= -1:
        m.eval()

#-------------------------------------------------------------
#The val function has some post-processing and drawing calling functions internally;val函数，内部有一些后处理和绘图调用功能
#-------------------------------------------------------------
def validate(val_loader, model, args, in_ave_gt = 0.5, in_rel_ave_gt = 1.0, txt_name = None):
    #Is it compared 是否比较
    #Drawing threshold parameters 画图阈值参数
    errorT = 0.1  #0.05

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_g = AverageMeter('LossG', ':.4e')
    losses_f = AverageMeter('LossF', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('Acc@2', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top10 = AverageMeter('Acc@10', ':6.2f')
    Rmae = AverageMeter('RMAE', ':6.3f')
    Rmse = AverageMeter('RMSE', ':6.3f')
    Rvar = AverageMeter('RVAR', ':6.3f')
    rel_top1 = AverageMeter('rel_Acc@1', ':6.2f')
    rel_top2 = AverageMeter('rel_Acc@2', ':6.2f')
    rel_top5 = AverageMeter('rel_Acc@5', ':6.2f')
    rel_top10 = AverageMeter('rel_Acc@10', ':6.2f')
    rel_Rmae = AverageMeter('rel_RMAE', ':6.3f')
    rel_Rmse = AverageMeter('rel_RMSE', ':6.3f')
    rel_Rvar = AverageMeter('rel_RVAR', ':6.3f')
    ave_time = AverageMeter('ave_Time', ':6.3f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [top1, top2, top5, top10, Rmae, Rmse, Rvar,  \
            rel_top1, rel_top2, rel_top5, rel_top10, rel_Rmae, rel_Rmse, rel_Rvar, ave_time],
        prefix='Test: ')
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        if txt_name != None:
            flabel =  open(str(txt_name) + 'label.txt', 'w')
            fresult =  open(str(txt_name) + 'result.txt', 'w')
            fpicname =  open(str(txt_name) + 'picname.txt', 'w')
            fdwl =  open(str(txt_name) + 'dwl.txt', 'w')
        for i, (images, target, image_name, cv_images, target_scale, dwl) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)
                target_scale = target_scale.cuda(args.gpu, non_blocking=True)
            ave_gt = torch.ones_like(target) *  in_ave_gt   ##autoext
            rel_ave_gt = torch.ones_like(target) *  in_rel_ave_gt
            # compute output
            ave_time_start = time.time()
            if True == args.use_amp: 
                with autocast():
                    output_g, output_f, y_base, target_f = model(images, target)
                    output = y_base  + args.CAR * output_f
            else:
                output_g, output_f, y_base, target_f = model(images, target)
                output = y_base  + args.CAR * output_f
         
            output[output > 1] = 1
            output[output < 0] = 0   
            ave_time.update(time.time()- ave_time_start, images.size(0))
            if txt_name != None:
                for j in range(len(target)):
                    print(image_name[j], target[j], output[j])
                    flabel.write(str(target[j].item()))
                    flabel.write('\n')
                    fresult.write(str(output[j].item()))
                    fresult.write('\n')
                    fpicname.write(image_name[j])
                    fpicname.write('\n')
                    fdwl.write(str(dwl[j].item()))
                    fdwl.write('\n')
            #Draw the result chart画结果图

            if True == args.use_amp: 
                output_g = output_g.to(torch.float32)

            # measure accuracy and record loss

            rel_output = output * target_scale
            rel_target = target * target_scale
            rel_output_g = output_g * target_scale

            acc1,acc2, acc5, acc10 = val_accuracy(output, target, acck=(0.01, 0.02, 0.05, 0.1))
            rel_acc1, rel_acc2, rel_acc5, rel_acc10 = val_accuracy(rel_output, rel_target, acck=(0.01, 0.02, 0.05, 0.1))

            
            mae = val_mae(output, target)
            mse = val_mse2(output, target)        
            var = val_mse2(ave_gt, target)  
            
            rel_mae = val_mae(rel_output, rel_target)
            rel_mse = val_mse2(rel_output, rel_target)          
            rel_var = val_mse2(rel_ave_gt, rel_target)  


            top1.update(acc1[0], images.size(0))
            top2.update(acc2[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            top10.update(acc10[0], images.size(0))
            Rmae.update(mae[0], images.size(0))
            Rmse.update(mse[0], images.size(0))
            Rvar.update(var[0], images.size(0))

            rel_top1.update(rel_acc1[0], images.size(0))
            rel_top2.update(rel_acc2[0], images.size(0))
            rel_top5.update(rel_acc5[0], images.size(0))
            rel_top10.update(rel_acc10[0], images.size(0))
            rel_Rmae.update(rel_mae[0], images.size(0))
            rel_Rmse.update(rel_mse[0], images.size(0))
            rel_Rvar.update(rel_var[0], images.size(0)) 
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if i % args.print_freq == 0:
                progress.display(i)

        if txt_name != None:
            flabel.close()
            fresult.close()   
            fpicname.close()      
        progress.display_summary()

    #print(model.module.final_layer_reg[4])


    return top1.avg,top2.avg, top5.avg, top10.avg,Rmae.avg, math.sqrt(Rmse.avg),  (1 - Rmse.avg / (Rvar.avg+0.000001)),\
            rel_top1.avg,rel_top2.avg,rel_top5.avg,rel_top10.avg,rel_Rmae.avg, math.sqrt(rel_Rmse.avg),  (1 - rel_Rmse.avg / (rel_Rvar.avg+0.000001)),\
            ave_time.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if is_best:
        torch.save(state, filename)
    #if is_best:
    #    shutil.copyfile(filename, 'model_best.pth.tar')

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

#Too slow, use torch.media later 太慢，后面使用torch.media
def get_median(data):
    tmp = sorted(data)
    half = len(tmp) // 2
    return (tmp[half] + tmp[~half]) / 2

#What is the correct value for acck, with a calculation accuracy of 0.05 corresponding to 10cm acck 以小于多少为正确，计算准确率 0.05对应10cm
def val_accuracy(output, target, acck=(0.05,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)
        wlstrip_size = target.size(1)
        res = []
        #The accuracy is also based on the median 准确率也以中值为准
        mid_output = output.quantile(q = 0.5, dim  = 1)
        mid_target = target.quantile(q = 0.5, dim  = 1)

        for k in acck:
            correct = (mid_output- mid_target).abs()  < k
            correct_k = correct.reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
#Measuring MSE 度量MSE
def val_mse2(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        wlstrip_size = target.size(1)    
        mid_output = output.quantile(q = 0.5, dim  = 1)
        mid_target = target.quantile(q = 0.5, dim  = 1)
        dis = (mid_output - mid_target) * 100.0
        dis = dis * dis
        dis = dis.reshape(-1).float().sum(0, keepdim=True)
        mse2 = dis / batch_size

        return mse2

#Measuring MAE 度量MAE
def val_mae(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        wlstrip_size = target.size(1)    
        mid_output = output.quantile(q = 0.5, dim  = 1)
        mid_target = target.quantile(q = 0.5, dim  = 1)
        dis = (mid_output - mid_target) * 100.0
        dis = dis.abs()
        dis = dis.reshape(-1).float().sum(0, keepdim=True)
        mae = dis / batch_size
        return mae
if __name__ == '__main__':
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
