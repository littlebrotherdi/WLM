import os, glob,shutil
import json
import numpy
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2,PIL
import random,math
import copy

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

#放__init__里只能一次，但是放__getitem__则可以做到增强的能力
init_process = transforms.Compose([
    #transforms.Scale(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),    
    normalize])

TT_process = transforms.Compose([transforms.ToTensor()])
NL_process = transforms.Compose([normalize])
RE_process = transforms.Compose([transforms.RandomErasing(p=0.5, 
    scale=(0.02, 0.05), ratio=(0.25, 0.75), value='random', inplace=False)])
'''
#1m
cut_process = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((1280, 64), padding = (12,0,12,0), padding_mode = 'reflect')
    ])
#2m

cut_process = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((1280, 64), padding = (12,0,12,0), padding_mode = 'reflect')
    ])
'''
color_process = transforms.Compose([
    transforms.ColorJitter(brightness = 0.3, contrast = 0.3, hue = 0.15, saturation = 0.3),
    transforms.RandomGrayscale(p = 0.4)
    ])


#实现随机水平翻转，PIL image
def WLG_random_horizontal_flip(img, pointsY = ([0] * 64) , p=0.5):
    #tmp_img =  copy.deepcopy(img)
    tmp_img = copy.deepcopy(img)
    tmp_pointsY = copy.deepcopy(pointsY)
    if random.random() < p:
        tmp_img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        tmp_pointsY = pointsY[::-1]
    
    return tmp_img, tmp_pointsY

#实现左右padding12个像素后（reflect）随机crop成原来大小 PIL image
def WLG_random_crop(img, pointsY = ([0] * 64), ex = 12 ):
    #w = img.size[0]
    #h = img.size[1]
    #ex = 12
    
    tmp_img = PIL.Image.new(img.mode, (img.size[0] + ex * 2, img.size[1]))
    tmp_list = [0] * (len(pointsY) + ex * 2)


    l_img = img.crop((0, 0, ex,  img.size[1])).transpose(PIL.Image.FLIP_LEFT_RIGHT)
    r_img = img.crop((img.size[0]-ex, 0, img.size[0],  img.size[1])).transpose(PIL.Image.FLIP_LEFT_RIGHT)
    tmp_img.paste(l_img,(0, 0))
    tmp_img.paste(img,(ex, 0))
    tmp_img.paste(r_img,(tmp_img.size[0]-ex, 0))


    tmp_list[ex:-ex] = pointsY
    tmp_list[:ex   ] = pointsY[ex-1 :: -1]
    tmp_list[-ex:  ] = pointsY[: -ex - 1: -1 ]  


    rint = random.randint(0,ex * 2)
    res_img = tmp_img.crop((rint, 0, rint+img.size[0],  img.size[1]))
    res_list = tmp_list[ rint: img.size[0]+rint]

    return res_img, res_list


##
def v_expand_image(pil_image , label):
    #新建2倍高度的尺子模版
    img_w = pil_image.size[0]
    img_h = pil_image.size[1]
    new_img = PIL.Image.new(pil_image.mode, (img_w, img_h * 2 * 2), (50,50,50))
    new_half_img = PIL.Image.new(pil_image.mode, (img_w, img_h * 2), (205,205,205))
    new_img.paste(new_half_img, (0, 0))
    #计算上下余量
    pri_eps= 0.000001

    up_h = round(max(0, label - 0.025) * img_h)
    dp_h = round(max(0, (1 - label) - 0.025)  * img_h)

    #扣取余量图片
    u_e = pil_image.crop((0, 0, img_w,  up_h))
    u_e_t = u_e.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    d_e = pil_image.crop((0, img_h - dp_h, img_w,  img_h))
    d_e_t = d_e.transpose(PIL.Image.FLIP_TOP_BOTTOM)


    #计算原图在大图中的上下坐标，并先贴上
    u_loc = round(2 * img_h - label * img_h) 
    d_loc = u_loc + img_h
    '''
    l_loc = img_w / 2
    r_loc = l_loc + img_W
    '''
    new_img.paste(pil_image,(0, u_loc))

    #贴图重复的次数
    u_iter = math.ceil(u_loc / (up_h + pri_eps))
    d_iter = math.ceil((4 * img_h - d_loc) / (dp_h + pri_eps)) 
    #循环贴图
    if u_iter < 100:
        for i in range(u_iter):
            if i % 2 == 0:
                new_img.paste(u_e_t, (0, u_loc - (i + 1) * up_h))
            else:
                new_img.paste(u_e, (0, u_loc - (i + 1) * up_h))
    else:
        print('u_iter',u_iter)
    if d_iter < 100:
        for i in range(d_iter):
            if i % 2 == 0:
                new_img.paste(d_e_t, (0, d_loc + i * dp_h))
            else:
                new_img.paste(d_e, (0, d_loc + i * dp_h))
    else:
        print('d_iter',d_iter)    
    '''
    l_e = new_img.crop((l_loc, 0, l_loc + img_w / 2,  4 * img_h))
    l_e_t = l_e.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    r_e = new_img.crop((l_loc + img_w / 2, 0, l_loc + img_w,  4 * img_h))
    r_e_t = r_e.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    new_img.paste(l_e_t, (0, 0))
    new_img.paste(r_e_t, (0, r_loc))
    '''

    return new_img

def scale_half_image(e_pil_image, wl, pointsY = ([0] * 64), scale = [0.5,2], dst_size = [64, 1280], r_scale = 0):
    ###截取 中心，然后resize

    #res_list = copy.deepcopy(pointsY)
    h_scale = 1
    w = dst_size[0]
    i_h = dst_size[1] * 4
    o_h = dst_size[1] * 2
    if 1 == r_scale:
        h_scale = random.uniform(scale[0], scale[1])
        #w = e_pil_image.size[0]
        #h = int(e_pil_image.size[1] * h_scale +0.5)   
        h = int(i_h * h_scale +0.5)
        re_img = e_pil_image.resize((w, h), PIL.Image.BILINEAR)

        #re_img = e_pil_image.resize((w, h))
        h_scale = h / (i_h)
    else:
        re_img = e_pil_image.resize((w, i_h))
    
    l_loc = 0
    t_loc = (re_img.size[1] - o_h) // 2

    dst_img = re_img.crop((l_loc, t_loc, w,  t_loc + o_h))

    res_list = [max(0, min(1, (x - wl) * h_scale + wl))  for x in pointsY]
    return dst_img, res_list




def get_median(data):
    tmp = sorted(data)
    half = len(tmp) // 2
    return (tmp[half] + tmp[~half]) / 2

def draw_data_demo(e_pil_image, water_level_f, water_sline, img_w, img_h, onlyflag, img_name, mfl_img_name, wl, now_wl, w, markly):

    top_loc = (1 - water_level_f) * img_h
    image = e_pil_image.crop((0, top_loc, img_w,  top_loc + img_h))
    image_arry = numpy.array(image)

    #画原图
    cv_image = cv2.cvtColor(image_arry, cv2.COLOR_RGB2BGR)
    out_path = './aug_tmp/{}_{}_{}_orlabel{:.4f}_nowlabel{:.4f}_w{:.2f}_{}.jpg'\
        .format(onlyflag, img_name, mfl_img_name, wl, now_wl, w, markly)
    cv2.imwrite(out_path,cv_image)
    
    #画line

    cv_image = cv2.cvtColor(image_arry, cv2.COLOR_RGB2BGR)
    cv2.line(cv_image, (0, int(water_level_f * img_h+0.5)), (img_w, int(water_level_f * img_h+0.5)), (0, 0, 255), 4)
    out_path = './aug_tmp/{}_{}_{}_orlabel{:.4f}_nowlabel{:.4f}_w{:.2f}_{}_line.jpg'\
        .format(onlyflag, img_name, mfl_img_name, wl, now_wl, w, markly)
    cv2.imwrite(out_path,cv_image)  

    #画sline
    cv_image = cv2.cvtColor(image_arry, cv2.COLOR_RGB2BGR)
    pts = []
    for i in range(img_w):
        pt = [i, round(water_sline[i] * img_h)]
        pts.append(pt)
    pts=numpy.array(pts, numpy.int32)
    cv2.polylines(cv_image, [pts], False, (255, 0, 255), 4)     # True表示该图形为封闭图形
    out_path = './aug_tmp/{}_{}_{}_orlabel{:.4f}_nowlabel{:.4f}_w{:.2f}_{}_sline.jpg'\
        .format(onlyflag, img_name, mfl_img_name, wl, now_wl, w, markly)
    cv2.imwrite(out_path,cv_image)  

class Wlg_Dataset(Dataset):



    def __init__(self, json_file_path, is_transform = False, ADA = 0, RED = 0, MFL = 0, MUS = 0, FFS = 0, RER = 0, img_size = [64, 640]):
        self._json_paths = glob.glob(json_file_path) #input
        self._imgs = []        #X
        #self._img_paths = []   #X
        self._water_level_f = [] #Y
        self._Dwater_level_f = [] #Y  连续对比时使用 记录偏差值
        self._water_level_scale = [] #归一化水位线到实际的位置
        self._water_sline = []
        self._image_name = []
        self._pil_image = []
        #self._cv_image = []
        self._is_transform = is_transform

        self._label_lenth = len(self._json_paths)
        self._e_pil_image = []
        self._e_water_sline = []
        self._ada = ADA #是否使用垂直延展的能力
        self._red = RED #是否使用mixup的能力   如果PE未开 mixup也不能开
        self._mfl = MFL #是否使用普通增强，包括裁剪和颜色
        self._mus = MUS #是否使用普通增强，包括裁剪和颜色
        self._ffuse = FFS
        self._rer = RER
        self._img_w = img_size[0]
        self._img_h = img_size[1]
        #self._distribute = [0,0,0,0,0,0,0,0,0,0]

        pri_eps= 0.000001

        for json_path_name in self._json_paths:
            with open(json_path_name) as f:
                (jsonpath, jsonname) = os.path.split(json_path_name)
                json_file = json.load(f)
                image_name = json_file['imagePath']
                image_full_path = jsonpath + '/' + image_name
                #print(image_full_path)
                pil_image = PIL.Image.open(image_full_path)
                water_sline = json_file['waterLevelStrip_f']
                if pil_image.size[0] != self._img_w  or  pil_image.size[1] != self._img_h:
                    
                    tmp_water_sline = []
                    for i in range(self._img_w):
                        or_loc = (i * pil_image.size[0]) / self._img_w
                        w = or_loc - int(or_loc)
                        if or_loc >= pil_image.size[0] - 1:
                            val = water_sline[pil_image.size[0] - 1]
                        else:
                            val = water_sline[int(or_loc)] * (1-w) + water_sline[int(or_loc) + 1] * w
                        tmp_water_sline.append(1.0 * val)
                    water_sline = tmp_water_sline          
                    #pil_image = pil_image.resize((self._img_w, self._img_h), PIL.Image.BILINEAR)   
                    pil_image = pil_image.resize((self._img_w, self._img_h))
                #water_loc = json_file['waterLevelLocal_f']
                
                
                water_level_f = json_file['waterLevelLocal_f']   #get_median(water_sline) #用stripline的中值做wl
                e_pil_image = v_expand_image(pil_image, water_level_f)
                water_level_scale = 10.0 / json_file['model_unit_length_f'] / 100  # 10cm  / 单个块的比例 /100cm/m = scale 单位米
                if 'CwaterLevelLocal_f' in json_file:
                    Dwater_level_f = json_file['CwaterLevelLocal_f'] - json_file['waterLevelLocal_f']   ##连续使用时记录差值 
                else:
                    Dwater_level_f = 0
                self._Dwater_level_f.append(Dwater_level_f)

                self._water_level_f.append(water_level_f)
                self._water_level_scale.append(water_level_scale)
                self._water_sline.append(water_sline)
                self._image_name.append(image_name)    
                #self._cv_image.append(cv_image)  
                self._pil_image.append(pil_image)  
                self._e_pil_image.append(e_pil_image)  

                '''   # 按照标注分类文件夹 1-10
                self._distribute[int(water_loc * 10)] += 1

                #将不同文件夹拷贝，有权限问题，compare补充了下
                f.close()
                
                if not os.path.isfile(json_path_name):
                    print('no json_path_name',json_path_name)
                if not os.path.isfile(image_full_path):
                    print('no image_full_path',image_full_path)


                if not os.path.exists(jsonpath + '/{}/'.format(int(water_loc * 10))):
                    os.makedirs(jsonpath + '/{}/'.format(int(water_loc * 10)))

                out_json_path = jsonpath + '/{}/'.format(int(water_loc * 10)) + jsonname
                out_img_path = jsonpath + '/{}/'.format(int(water_loc * 10)) + image_name
                shutil.copyfile(json_path_name, out_json_path)
                shutil.copyfile(image_full_path, out_img_path)
                '''

    def __len__(self):
        return self._label_lenth

    def __getitem__(self, idx):
        use_PE = self._red  #是否使用垂直延展的能力
        use_mixup = self._mfl and random.randint(0, 1)   #是否使用mixup的能力   如果PE未开 mixup也不能开 50%开
        use_aug = self._ada #是否使用普通增强，包括裁剪和颜色
        use_mus = self._mus #是否使用普通增强，包括裁剪和颜色
        use_flipfuse = self._ffuse
        use_rer = self._rer
        data_aug_demo = 0

        water_level_f = self._water_level_f[idx]
        Dwater_level_f = self._Dwater_level_f[idx]
        water_level_scale = self._water_level_scale[idx]
        water_sline = self._water_sline[idx]
        image_name = self._image_name[idx] 
        pil_image = self._pil_image[idx] # 只是原始信息
        img_w = self._img_w
        img_h = self._img_h
        or_water_level_f = water_level_f
        onlyflag = random.randint(0,999999)

        image_name2 = None
        or_water_level_f2 = -10
        r_w = -10

        #e_pil_image = self._e_pil_image[idx]
        ###scale use_mus
        e_pil_image, e_water_sline = scale_half_image(self._e_pil_image[idx], water_level_f, 
            self._water_sline[idx], dst_size = [self._img_w, self._img_h],  r_scale = 0)


        if 1 == data_aug_demo:
      
            #主图原图
            pil_image_arry = numpy.array(pil_image)
            cv_image = cv2.cvtColor(pil_image_arry, cv2.COLOR_RGB2BGR)
            #主图名字，当前图名字,label，改变后的label，权重
            out_path = './aug_tmp/{}_{}_{}_orlabel{:.4f}_nowlabel{:.4f}_w{:.2f}.jpg'\
                .format(onlyflag, image_name, image_name, or_water_level_f, water_level_f, 1.0)
            cv2.imwrite(out_path,cv_image)  

 
        if self._is_transform is True:

        #mixup 如果mixup 初始化2号图像
            if 1 == use_mixup and 1 == use_PE:
                r_idx = random.randint(0, self._label_lenth - 1)
                r_w = random.uniform(0.0, 0.5)
            #r_w = 0.375  
                water_level_f2 = self._water_level_f[r_idx]
                water_sline2 = self._water_sline[r_idx]
                image_name2 = self._image_name[r_idx]  
                pil_image2 = self._pil_image[r_idx] # 只是原始信息
                or_water_level_f2 = water_level_f2
                e_pil_image2, e_water_sline2 = scale_half_image(self._e_pil_image[r_idx], water_level_f2, 
                    self._water_sline[r_idx], dst_size = [self._img_w, self._img_h], r_scale = 0)
            #e_pil_image2 = self._e_pil_image[r_idx]
            

                if 1 == data_aug_demo:

                #附图原图
                    pil_image_arry2 = numpy.array(pil_image2)
                    cv_image2 = cv2.cvtColor(pil_image_arry2, cv2.COLOR_RGB2BGR)
                #附图名字，当前图名字,label，改变后的label，权重
                    out_path2 = './aug_tmp/{}_{}_{}_orlabel{:.4f}_nowlabel{:.4f}_w{:.2f}.jpg'\
                        .format(onlyflag, image_name, image_name2, or_water_level_f2, water_level_f2, 1.0)
                    cv2.imwrite(out_path2,cv_image2)  



            #新的随机label
            if 1 == use_mus:

                e_pil_image, e_water_sline = scale_half_image(self._e_pil_image[idx], water_level_f, 
                    self._water_sline[idx], dst_size = [self._img_w, self._img_h],  r_scale = use_mus)
                if 1 == data_aug_demo:
                    draw_data_demo(e_pil_image, water_level_f, e_water_sline, img_w, img_h,
                        onlyflag, image_name, image_name, water_level_f, water_level_f, 1-r_w, 'or1')

                if 1 == use_mixup and 1 == use_PE:
                    e_pil_image2, e_water_sline2 = scale_half_image(self._e_pil_image[r_idx], water_level_f2, 
                        self._water_sline[r_idx], dst_size = [self._img_w, self._img_h],  r_scale = use_mus)
                    if 1 == data_aug_demo:
                        draw_data_demo(e_pil_image2, water_level_f2, e_water_sline2, img_w, img_h,
                            onlyflag, image_name, image_name2, water_level_f, water_level_f2, r_w, 'or2')

            if 1 == use_aug:
                #e_pil_image = cut_process(e_pil_image)
                
                e_pil_image, e_water_sline  = WLG_random_horizontal_flip(e_pil_image, e_water_sline)
                e_pil_image, e_water_sline = WLG_random_crop(e_pil_image, e_water_sline, ex = int(0.25 * self._img_w +0.5))
                e_pil_image  = color_process(e_pil_image)  

                if 1 == data_aug_demo:

                    draw_data_demo(e_pil_image, water_level_f, e_water_sline, img_w, img_h,
                        onlyflag, image_name, image_name, or_water_level_f, water_level_f, 1-r_w, 'ADA1')


                if 1 == use_mixup and 1 == use_PE:
                    #e_pil_image2 = cut_process(e_pil_image2)
                    e_pil_image2, e_water_sline2 = WLG_random_horizontal_flip(e_pil_image2, e_water_sline2)
                    e_pil_image2, e_water_sline2 = WLG_random_crop(e_pil_image2, e_water_sline2, ex = int(0.25 * self._img_w +0.5))
                    e_pil_image2  = color_process(e_pil_image2)  

                    if 1 == data_aug_demo:
                        draw_data_demo(e_pil_image2, water_level_f2, e_water_sline2, img_w, img_h,
                            onlyflag, image_name, image_name2, or_water_level_f2, water_level_f2, r_w, 'ADA2')

        
            if 1 == use_PE:
                '''
                tmp_r = random.uniform(0.00, 1.00)#
                if tmp_r > 1.0 / 6:
                    pe_level = random.uniform(0.25, 0.75)#20220612下次干这个
                elif tmp_r > 1.0 / 12 and tmp_r <= 1.0 / 6:
                    pe_level = random.uniform(0.75, 1.00)
                elif tmp_r <= 1.0 / 12:
                    pe_level = random.uniform(0.00, 0.25)
                else:
                    print('random error')
                '''
                pe_level = random.uniform(0.00001, 0.99999)#20220612下次干这个
                e_water_sline = [i - (water_level_f - pe_level) for i in e_water_sline]
                water_level_f = pe_level
                label_loc = (1 - water_level_f) * img_h
                #image = e_pil_image.crop((0, label_loc, img_w,  label_loc + img_h))

                if 1 == data_aug_demo:
                    draw_data_demo(e_pil_image, water_level_f, e_water_sline, img_w, img_h,
                        onlyflag, image_name, image_name, or_water_level_f, water_level_f, 1-r_w, 'ADA+PED1')


                if 1 == use_mixup and 1 == use_PE:
                    e_water_sline2 = [i - (water_level_f2 - pe_level) for i in e_water_sline2]
                    water_level_f2 = pe_level
                    label_loc2 = (1 - water_level_f2) * img_h
                    #image2 = e_pil_image2.crop((0, label_loc2, img_w,  label_loc2 + img_h))


                    # 直接融合  
                    e_pil_image = PIL.Image.blend(e_pil_image, e_pil_image2, r_w)               
                    for i in range(len(water_sline)):
                        e_water_sline[i] = e_water_sline[i]* (1 - r_w) + e_water_sline2[i]* r_w

                    if 1 == data_aug_demo:
                        draw_data_demo(e_pil_image2, water_level_f2, e_water_sline2, img_w, img_h,
                            onlyflag, image_name, image_name2, or_water_level_f2, water_level_f2, r_w, 'ADA+PED2') 

                        draw_data_demo(e_pil_image, water_level_f, e_water_sline, img_w, img_h,
                            onlyflag, image_name, image_name2, or_water_level_f, water_level_f, 1-r_w, 'ADA+PED+MFL')    
                         
            if 1 == use_flipfuse:

                re_e_pil_image = e_pil_image.transpose(PIL.Image.FLIP_TOP_BOTTOM)

                re_e_pil_image, _  = WLG_random_horizontal_flip(re_e_pil_image, e_water_sline)
                re_e_pil_image, _ = WLG_random_crop(re_e_pil_image, e_water_sline, ex = int(0.25 * self._img_w + 0.5))
                #re_e_pil_image  = color_process(re_e_pil_image)  
        
                r_f = random.uniform(0.0, 0.3)
                e_pil_image = PIL.Image.blend(e_pil_image, re_e_pil_image, r_f)   
            
        label_loc = round((1 - water_level_f) * img_h)
        d_wl = ((1 - water_level_f) * img_h - label_loc) / img_h

        image = e_pil_image.crop((0, label_loc, img_w,  label_loc + img_h))                   
        

        #init_image = init_process(image)  
        init_image = TT_process(image)     
        if 1 == use_rer:
            init_image = RE_process(init_image)
            init_image = RE_process(init_image)
            image = transforms.ToPILImage()(init_image).convert('RGB')

        image_arry = numpy.array(image)
        cv_image = cv2.cvtColor(image_arry, cv2.COLOR_RGB2BGR)


        init_image = NL_process(init_image)



        sline = torch.tensor([e_water_sline], dtype = float)

        if 1 == data_aug_demo:
            pts = []
            for i in range(img_w):
                pt = [i, round(e_water_sline[i] * img_h)]
                pts.append(pt)
            pts=numpy.array(pts, numpy.int32)
            cv2.polylines(cv_image, [pts], False, (255, 0, 255), 4)     # True表示该图形为封闭图形
            out_path = './aug_tmp/{}_{}_{}_orlabel{:.4f}_nowlabel{:.4f}_w{:.2f}_{}.jpg'\
                .format(onlyflag, image_name, image_name2, or_water_level_f, water_level_f, r_w, 'final_out')
            cv2.imwrite(out_path,cv_image) 
        #sline = e_water_sline[16:49:8]
        sline = sline.quantile(q = 0.5, dim = 1)
        sline = sline + d_wl
        water_level_f = torch.tensor([water_level_f]) + d_wl
        #sline = torch.tensor(sline)
        return init_image, sline, image_name, cv_image, water_level_scale, Dwater_level_f



#if __name__ == '__main__':
#    main()
