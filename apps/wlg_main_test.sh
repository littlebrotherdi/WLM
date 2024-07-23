#!/usr/bin/env bash
clear 
DATA='../data/mapping_val/'
# RESUME=''

#RESUME='/data1/private/Project_branch/1.wlg/wlg_reg/apps/aug-ped-mixup/CP_hrnetW18-Aug-1M0613_ep616_a1_52.04_a2_71.95_a5_88.69_a10_93.21_rmse_0.059803_aT_0.171168.pth.tar'
#RESUME='/data1/private/Project_branch/1.wlg/wlg_reg/apps/aug-ped-mixup/CP_hrnetW18-Aug-PDE-mixup-1M_ep1019_a1_54.30_a2_83.71_a5_90.95_a10_94.12_rmse_0.073253.pth.tar'


#RESUME='/data1/private/Project_branch/1.wlg/wlg_reg/apps/aug-ped-mixup/CP_hrnetW18-Aug-PDE-mixup-mixup-1M0613_ep1040_a1_60.63_a2_85.52_a5_91.86_a10_94.57_rmse_0.073630.pth.tar'
RESUME='./output/CPMB_BASECAR0.20_2151_w64h640Cw64Ch640modelhrnet_w18sv2_3_w18sv2_3ep140ForG1acc76.382_86.131_93.266_96.281rmae1.5845rmse4.7604r20.9393aT0.2866.pth.tar'
#RESUME='/data1/private/Project_branch/1.wlg/wlg_reg/apps/20220708/CP_model_hrnet18_sample_ADA_1_RED_0_MFL_0_ep558_a1_40.72_a2_66.52_a5_81.90_a10_89.14_rmse_0.082662_aT_0.145874.pth.tar'
#RESUME='/data1/private/Project_branch/1.wlg/wlg_reg/apps/20220706/CP_AlexNet-1M_ep100_a1_29.86_a2_49.32_a5_72.40_a10_85.97_rmse_0.085615_aT_0.006374.pth.tar'
echo "DATA1EXT = $DATA"
echo "RESUME = $RESUME"

python test.py $DATA --arch hrnet_w18sv2_3_w18sv2_3 --img_w 64 --img_h 640 --cas_img_w 64 --cas_img_h 640  --CAR 0.2\
     --resume $RESUME \
     -e 