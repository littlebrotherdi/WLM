# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
import copy, random,math

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(False)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i], 
                                       momentum=BN_MOMENTUM),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3, 
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(HighResolutionNet, self).__init__()
        self.self_counter = 0
        self.cfg = cfg
        self.cascade_ratio = cfg['MODEL']['CASCADE_RATIO']
        self.half_cara = 0.5 * self.cascade_ratio
        self.quar_cara = 0.25 * self.cascade_ratio
        self.img_sz = self.cfg['MODEL']['IMAGE_SIZE']
        self.cascade_img_sz = self.cfg['MODEL']['CASCADE_IMAGE_SIZE']  ###外面要配置
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        #原图1     branch 1,2,3,4     4 8 16 32
        #下采样取整对齐倍数
        self.round_base = 32
        #self.round_list = [8, 4, 2, 1]
        self.round_list = [1, 1, 1, 1]

        ##Module1
        self.stage1_cfg = cfg['MODEL']['EXTRA']['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, stage2_out_channel = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            stage2_out_channel, num_channels)
        self.stage3, stage3_out_channel = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            stage3_out_channel, num_channels)
        self.stage4, stage4_out_channel = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)
        
        stage4_out_channel_sum = np.int(np.sum(stage4_out_channel))
        self.final_layer_reg = nn.Sequential(
            nn.Conv2d(
                in_channels=stage4_out_channel_sum,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(96, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=96,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(1, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
               

        # Classification Head
        #self.final_layer1 = self._make_liner(pre_stage_channels, 240, 1024)#[4 8 16 32]
        #self.final_layer2 = self._make_liner(pre_stage_channels, 1024, 256)#[4 8 16 32]
        #self.final_layer3 = self._make_liner(pre_stage_channels, 256, 1)#[4 8 16 32]
       
        #1m
        self.classifier = nn.Sequential(
            nn.Linear(int(self.img_sz[0] * self.img_sz[1] / 4 / 4 + 0.5), 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1)
        )
        #2m
        '''
        self.classifier = nn.Sequential(
            nn.Linear(5120, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1)
        )
        '''

        ##Module2-brachs  需要修改成分辨率相关的

        self.b_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=(2,2), padding=1,
                               bias=False)
        self.b_bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)

        self.b_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.b_bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)

        #只用一层
        self.b_stage1_cfg = cfg['MODEL']['EXTRA']['B_STAGE1']
        b_num_channels = self.b_stage1_cfg['NUM_CHANNELS'][0]
        b_block = blocks_dict[self.b_stage1_cfg['BLOCK']]
        b_num_blocks = self.b_stage1_cfg['NUM_BLOCKS'][0]
        self.b_layer1 = self._make_layer(b_block, 64, b_num_channels, b_num_blocks)
        b_stage1_out_channel = b_block.expansion*b_num_channels
        
        
        #stage1_out_channel_sum = np.int(np.sum(stage1_out_channel))  ##融合主干通道      
        self.b_stage2_cfg = cfg['MODEL']['EXTRA']['B_STAGE2']
        b_num_channels = self.b_stage2_cfg['NUM_CHANNELS']
        b_block = blocks_dict[self.b_stage2_cfg['BLOCK']]
        b_num_channels = [
            b_num_channels[i] * block.expansion for i in range(len(b_num_channels))]
        #这个地方要增加主干通道
        #transition1 太浅 不做融合

        self.branch_fuse_layer_list1 = self._make_barnch_fuse_layer([stage1_out_channel], [b_stage1_out_channel])


        self.b_transition1 = self._make_transition_layer(
            [b_stage1_out_channel], b_num_channels)
        self.b_stage2, b_stage2_out_channel = self._make_stage(
            self.b_stage2_cfg, b_num_channels)

    
        #stage2_out_channel_sum = np.int(np.sum(stage2_out_channel))  ##融合主干通道
        self.b_stage3_cfg = cfg['MODEL']['EXTRA']['B_STAGE3']
        b_num_channels = self.b_stage3_cfg['NUM_CHANNELS']
        b_block = blocks_dict[self.b_stage3_cfg['BLOCK']]
        b_num_channels = [
            b_num_channels[i] * block.expansion for i in range(len(b_num_channels))]
        

        self.branch_fuse_layer_list2 = self._make_barnch_fuse_layer(stage2_out_channel, b_stage2_out_channel)

        self.b_transition2 = self._make_transition_layer(
            b_stage2_out_channel, b_num_channels)
 
        
        #这个地方要增加主干通道
        self.b_stage3, b_stage3_out_channel = self._make_stage(
            self.b_stage3_cfg, b_num_channels)
            
        stage3_out_channel_sum = np.int(np.sum(stage3_out_channel))  ##融合主干通道
        self.b_stage4_cfg = cfg['MODEL']['EXTRA']['B_STAGE4']
        b_num_channels = self.b_stage4_cfg['NUM_CHANNELS']
        b_block = blocks_dict[self.b_stage4_cfg['BLOCK']]
        b_num_channels = [
            b_num_channels[i] * block.expansion for i in range(len(b_num_channels))]
        

        self.branch_fuse_layer_list3 = self._make_barnch_fuse_layer(stage3_out_channel, b_stage3_out_channel)

        self.b_transition3 = self._make_transition_layer(
            b_stage3_out_channel, b_num_channels)
        
        self.b_stage4, b_stage4_out_channel = self._make_stage(
            self.b_stage4_cfg, b_num_channels, multi_scale_output=True)

        self.branch_fuse_layer_list4 = self._make_barnch_fuse_layer(stage4_out_channel, b_stage4_out_channel)

        b_stage4_out_channel_sum = np.int(np.sum(b_stage4_out_channel))
       
        self.b_final_layer_reg = nn.Sequential(
            nn.Conv2d(
                in_channels=b_stage4_out_channel_sum,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(96, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=96,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(1, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
               
       
        #1m
        self.b_classifier = nn.Sequential(
            nn.Linear(int(self.cascade_img_sz[0] * self.cascade_img_sz[1] / 4 / 4 + 0.5), 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1)
        )

    def _make_liner(self, pre_stage_channels, hin_channels, hout_channels):

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=hin_channels,
                out_channels=hout_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(hout_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        return  final_layer

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_barnch_fuse_layer(
            self, num_channels_list_m, num_channels_list_s):
        branch_fuse_layer_list = []
        for i, num_channals_s in enumerate(num_channels_list_s):
            fuse_layer = []
            for j, num_channals_m in enumerate(num_channels_list_m):
                fuse_layer.append(nn.Sequential(
                    nn.Conv2d(num_channals_m,
                                num_channals_s,
                                1,
                                1,
                                0,
                                bias=False),
                    nn.BatchNorm2d(num_channals_s, 
                                    momentum=BN_MOMENTUM)))
            branch_fuse_layer_list.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(branch_fuse_layer_list)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels


    def _process_branch_fuse_layer(
            self, branch_fuse_layer_list, m_feature_list, s_feature_list):
        out_feature_list = []
        for i, s_feature in enumerate(s_feature_list):
            s_h, s_w = s_feature.size(2), s_feature.size(3) 
            out_feature = s_feature
            for j, m_feature in enumerate(m_feature_list):
                m_feature_out = branch_fuse_layer_list[i][j](m_feature)
                #x = F.interpolate(m_feature_out, size=(s_h, s_w), mode='bilinear', align_corners=True)
                #x = F.interpolate(m_feature_out, size=(s_h, s_w), mode='bilinear')
                x = F.interpolate(m_feature_out, size=(s_h, s_w))
                out_feature = out_feature + x
            out_feature_list.append(out_feature)
        return out_feature_list

####需要增加参数 分辨率 扩展比例
    def _feature_cutout(self, in_feature, refer, target, ratio = 0.5, prc_model = 0, round_num = 1):  #prc_model, 0是根据结果随机，1是使用结果
        
        device = refer.device
        h = in_feature.shape[-2]
        y = []
        base_l = []
        target_f = []
        marge_r = 0.05 #ratio * 0.1        取值范围（0， 0.5*（ratio-round_num / h））  0.2 -0-0.075, 0.5-
        ##round_num / h
        for i in range(in_feature.shape[0]):
            gm = 0.5 * math.exp(-4*abs(target[i][0].item() - refer[i][0].item()))
            rd = random.random() 
            if 0 == prc_model and rd < gm:
                up   = max(0, min(1 - ratio, target[i][0].item() - ratio + marge_r + round_num / h))
                down = max(0, min(1 - ratio, target[i][0].item() - marge_r))
                h1 = int((random.uniform(up, down) * h + 0.5 * round_num) // round_num * round_num +0.5)
            else:
                h1 = int(((refer[i][0].item() - 0.5 * ratio) * h + 0.5 * round_num) // round_num * round_num +0.5)
            dh = int((ratio * h + 0.5 * round_num) // round_num * round_num +0.5)
            h2 = h1 + dh
            
            ###图像范围可能超出边界，ybase的范围为 -ratio*h ~ h，图像(-r * h - 0)~(h + r *h)
            if h1 < 0:
                y_t = F.pad(in_feature[i, :, 0 : max(1,h2), :].unsqueeze(0),(0,0, min(-h1, dh - 1),0), mode = 'replicate')
            elif h2 >= h:   
                y_t = F.pad(in_feature[i, :, min(h1, h-1) : h, :].unsqueeze(0),(0,0, 0, min(h2-h, dh - 1)), mode = 'replicate')
            else:
                y_t = in_feature[i, :, h1 : h2, :].unsqueeze(0)
            y.append(y_t)
                      
            bl = 1.0 * h1 / h  # base 在原图中的位置
            base_l.append(torch.tensor([bl ]).unsqueeze(0))

            tf = float(max(0, min(1, (target[i] * h - h1) / dh)))
            target_f.append(torch.tensor([tf]).unsqueeze(0))
        y = torch.cat(y, dim = 0)
        base_l = torch.cat(base_l, dim = 0).to(device).requires_grad_()
        target_f = torch.cat(target_f, dim = 0).to(device).requires_grad_()
        
        return y, base_l, target_f
    
        
    def forward(self, x, target = torch.zeros(1,1)):
        #print('input', x.size())
        x_b = x
        #x = F.interpolate(x, size=(self.img_sz[1], self.img_sz[0]), mode='bilinear', align_corners=True)
        #x = F.interpolate(x, size=(self.img_sz[1], self.img_sz[0]), mode='bilinear')
        x = F.interpolate(x, size=(self.img_sz[1], self.img_sz[0]))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        #print('basic', x.size())
        y_list_stage1 = [x]
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
            #print('stage2_', i , x_list[i].size())
        y_list_stage2 = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list_stage2[-1]))
            else:
                x_list.append(y_list_stage2[i])
            #print('stage3_', i , x_list[i].size())
        y_list_stage3 = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list_stage3[-1]))
            else:
                x_list.append(y_list_stage3[i])
            #print('stage4_', i , x_list[i].size())
        y_list_stage4 = self.stage4(x_list)

        #upsample,but not ori downsample
        y = y_list_stage4[0]
        x0_h, x0_w = y_list_stage4[0].size(2), y_list_stage4[0].size(3)  
        for i in range(1, self.stage4_cfg['NUM_BRANCHES']):   
            #x = F.interpolate(y_list_stage4[i], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
            #x = F.interpolate(y_list_stage4[i], size=(x0_h, x0_w), mode='bilinear')
            x = F.interpolate(y_list_stage4[i], size=(x0_h, x0_w))
            y = torch.cat([y, x], 1)   
            
        y = self.final_layer_reg(y)
        y = y.view(y.size(0), -1)
        y_gross = self.classifier(y)

        ##这里需要输入图像是 b分支处理的整数倍！！！！
        #round_num_or_b = self.round_base * int(x_b.size(2) / self.cascade_img_sz[1] + 0.5)
        round_num_or_b = 1
        
        if True == self.training:
            #rd = random.random()
            #if rd > 1.5:
            #    x, y_base, target_f  = self._feature_cutout(x_b, y_gross, target, self.cascade_ratio, prc_model = 0, round_num = round_num_or_b)
            #else:
            #    x, y_base, target_f  = self._feature_cutout(x_b, y_gross, target, self.cascade_ratio, prc_model = 1, round_num = round_num_or_b)
            x, y_base, target_f  = self._feature_cutout(x_b, y_gross, target, self.cascade_ratio, prc_model = 1, round_num = round_num_or_b)
        else:            
            y_gross[y_gross > 0.9999] = 0.9999
            y_gross[y_gross < 0.0001] = 0.0001  
            x, y_base, target_f  = self._feature_cutout(x_b, y_gross, target, self.cascade_ratio, prc_model = 1, round_num = round_num_or_b)
        y_mid = y_base + self.half_cara
        #x = F.interpolate(x, size=(self.cascade_img_sz[1], self.cascade_img_sz[0]), mode='bilinear', align_corners=True)
        #x = F.interpolate(x, size=(self.cascade_img_sz[1], self.cascade_img_sz[0]), mode='bilinear')
        x = F.interpolate(x, size=(self.cascade_img_sz[1], self.cascade_img_sz[0]))
        x = self.b_conv1(x)
        x = self.b_bn1(x)
        x = self.relu(x)       
        x = self.b_conv2(x)
        x = self.b_bn2(x)
        x = self.relu(x)
        x = self.b_layer1(x)

        #融合b_stage2和 stage2
        y_list = [x]
        x_fc_list = []  
        for i in range(self.stage1_cfg['NUM_BRANCHES']):
            x_c, y_base_t, target_f_t  = self._feature_cutout(y_list_stage1[i], y_mid, \
             target, self.cascade_ratio, prc_model = 1, round_num = self.round_list[i])
            x_fc_list.append(x_c)
        y_list = self._process_branch_fuse_layer(self.branch_fuse_layer_list1, x_fc_list, y_list)

        #b_stafe2推理
        x_list = []
        for i in range(self.b_stage2_cfg['NUM_BRANCHES']):
            if self.b_transition1[i] is not None:
                x_list.append(self.b_transition1[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.b_stage2(x_list)
        
        #融合b_stage2和 stage2
        x_fc_list = []  
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            x_c, y_base_t, target_f_t  = self._feature_cutout(y_list_stage2[i], y_mid, \
             target, self.cascade_ratio, prc_model = 1, round_num = self.round_list[i])
            x_fc_list.append(x_c)
        y_list = self._process_branch_fuse_layer(self.branch_fuse_layer_list2, x_fc_list, y_list)  
            
        #b_stafe3推理
        x_list = []
        for i in range(self.b_stage3_cfg['NUM_BRANCHES']):
            if self.b_transition2[i] is not None:
                x_list.append(self.b_transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.b_stage3(x_list)

        #融合b_stage3_out和 stage3_out_car
        x_fc_list = []  
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            x_c, y_base_t, target_f_t  = self._feature_cutout(y_list_stage3[i], y_mid, \
             target, self.cascade_ratio, prc_model = 1, round_num = self.round_list[i])
            x_fc_list.append(x_c)
        y_list = self._process_branch_fuse_layer(self.branch_fuse_layer_list3, x_fc_list, y_list)       
   
        #b_stafe4推理
        x_list = []
        for i in range(self.b_stage4_cfg['NUM_BRANCHES']):
            if self.b_transition3[i] is not None:
                x_list.append(self.b_transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.b_stage4(x_list)

        #融合b_stage4_out和 stage4_out_car
        x_fc_list = []  
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            x_c, y_base_t, target_f_t  = self._feature_cutout(y_list_stage4[i], y_mid, \
             target, self.cascade_ratio, prc_model = 1, round_num = self.round_list[i])
            x_fc_list.append(x_c)
        y_list = self._process_branch_fuse_layer(self.branch_fuse_layer_list4, x_fc_list, y_list)           


        #out  b_stage4
        y = y_list[0]
        x0_h, x0_w = y_list[0].size(2), y_list[0].size(3)  
        for i in range(1, self.b_stage4_cfg['NUM_BRANCHES']):   
            #x = F.interpolate(y_list[i], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
            #x = F.interpolate(y_list[i], size=(x0_h, x0_w), mode='bilinear')
            x = F.interpolate(y_list[i], size=(x0_h, x0_w))
            y = torch.cat([y, x], 1)   
        
        y = self.b_final_layer_reg(y)
        y = y.view(y.size(0), -1)
        y_fine = self.b_classifier(y)

        if False == self.training:
            y_fine[y_fine > 1] = 1.0
            y_fine[y_fine < 0] = 0.0

        return y_gross, y_fine, y_base, target_f

    def init_weights(self, pretrained='',):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


def get_cls_net(config, **kwargs):
    model = HighResolutionNet(config, **kwargs)
    model.init_weights()
    return model
