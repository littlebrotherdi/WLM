import os,sys, glob
import json
import cv2
import numpy as np


def get_max_min_value(martix):
  '''
  得到矩阵中每一列最大的值
  '''
  max_list=[]
  min_list = []
  for j in range(len(martix[0])):
    tmp_list=[]
    for i in range(len(martix)):
      tmp_list.append(martix[i][j])
    max_list.append(max(tmp_list))
    min_list.append(min(tmp_list))
  return max_list, min_list

def get_median(data):
    tmp = sorted(data)
    half = len(tmp) // 2
    return (tmp[half] + tmp[~half]) / 2
    

def read_json(file_path, out_path, fix_linelist = None):
    out_json = {}
    out_fix_linelist = None 
    with open(file_path) as f:
        json_inner  = json.load(f)
        image_name  = json_inner['imagePath']
        (jsonpath, jsonname) = os.path.split(file_path)

        img = cv2.imread(jsonpath + '/' + image_name)
        print(jsonpath + '/' + image_name)
        if len(jsonpath + '/' + image_name) == 0:
            sys.exit(0)
        img_w = json_inner['imageWidth']
        img_h = json_inner['imageHeight']
        for shape in json_inner['shapes']:
            if shape['label'] == "ROI" :
                rec_points = np.array(shape['points'])
                ROI_points = shape['points']
                ROI_bottom = max(ROI_points[0][1],  ROI_points[1][1]) - 5
                #print(ROI_bottom)

                rec_margn = 10
                wlg_model_unit_length_x = 64    #每个单位刻度的像素数 长度 = 宽度 
                wlg_model_unit_length_y_base = 640    #每个单位刻度的像素数 长度 = 宽度 
                wlg_model_unit_length_y = wlg_model_unit_length_y_base
                wlg_model_total_length =  10 * wlg_model_unit_length_y   #水尺总长 2m * 10个刻度 * 单位刻度像素数
                
                
                linelist = []
                wl_line =[]
                wl_strip= []
                # 找到同一个roi下的 标定线 和水位线
                for shape_inner in json_inner['shapes'] :
                    if shape_inner['shape_type'] == "line":
                        line_points = shape_inner['points']
                        if line_points[0][0] > rec_points.min(0)[0] - rec_margn and line_points[0][0] < rec_points.max(0)[0] + rec_margn \
                           and line_points[1][0] > rec_points.min(0)[0] - rec_margn and line_points[1][0] < rec_points.max(0)[0] + rec_margn\
                           and line_points[0][1] > rec_points.min(0)[1] - rec_margn and line_points[0][1] < rec_points.max(0)[1] + rec_margn\
                           and line_points[1][1] > rec_points.min(0)[1] - rec_margn and line_points[1][1] < rec_points.max(0)[1] + rec_margn:
                           if shape_inner['label'] == 'wl':
                                wl_line.append(shape_inner)
                           else : 
                                linelist.append(shape_inner)
                    if shape_inner['shape_type'] == "linestrip":
                        line_points = shape_inner['points']

                        max_lineP, min_lineP = get_max_min_value(line_points)

                        if max_lineP[0] < rec_points.max(0)[0] + rec_margn \
                            and max_lineP[1] < rec_points.max(0)[1] + rec_margn \
                            and min_lineP[0] > rec_points.min(0)[0] - rec_margn \
                            and min_lineP[1] > rec_points.min(0)[1] - rec_margn:
                            wl_strip.append(shape_inner)
                    

                p_num = len(linelist)
                if p_num < 3:
                    print('less than 3 height points')


                ## line的标注从上到下排列
                linelist = sorted(linelist, key = lambda k: (float(k.get('label'))), reverse = True)
                #print(linelist)
                '''
                [y;W] = [[a1, a2]; [a3, a4=1]] [Y;1]     Y原画面高度  h->y label

                '''
                left = []
                right = []
                h_base = float(linelist[0]['label'])
                ## 取3个点 以及其label
                Y_top = (linelist[0]['points'][0][1] + linelist[0]['points'][1][1]) * 0.5
                h_top = h_base - float(linelist[0]['label']) 
                left.append([Y_top, 1, -h_top*Y_top])
                right.append(h_top) 


                Y_mid = (linelist[p_num//2]['points'][0][1] + linelist[p_num//2]['points'][1][1]) * 0.5
                h_mid = h_base - float(linelist[p_num//2]['label'])
                left.append([Y_mid, 1, -h_mid*Y_mid])
                right.append(h_mid) 

                Y_end = (linelist[-1]['points'][0][1] + linelist[-1]['points'][1][1]) * 0.5
                h_end = h_base - float(linelist[-1]['label'])
                left.append([Y_end, 1, -h_end*Y_end])
                right.append(h_end)

                lineA = np.dot(np.linalg.inv(left), right)

                Y_bot = ROI_bottom

                y_bot = lineA[0] * Y_bot + lineA[1]
                w_bot = lineA[2] * Y_bot + 1

                h_bot = h_base - 1.0 * y_bot / w_bot
                #print(h_bot)

                ##后面根据实际高度需要换算比例尺

                wlg_model_unit_length_y = wlg_model_total_length / ((1.0 * y_bot / w_bot ))
                #print(wlg_model_unit_length_y)
                if wlg_model_unit_length_y < 0:
                    wlg_model_unit_length_y = wlg_model_unit_length_y_base


                
                ## 先算当前样本的量程 以及wlg_model_unit_length_y = 640 #每个单位刻度的像素数



                ###计算对应点
                
                ##原图点列表 和 模版图列表
                src_plist = []
                dst_plist = []
                highlist_locH = float(linelist[0]['label'])

                #计算各标定线位置及刻度 计算 对应 在 模版位置
                line_shape = linelist[0]
                src_p = sorted(line_shape['points'], key = lambda k : k[0])
                dst_p0_x = wlg_model_unit_length_x / 4
                #dst_p0_x = 0
                dst_p0_y = 0
                src_plist.append(src_p[0])
                dst_plist.append([dst_p0_x, dst_p0_y])

                #dst_p1_x = wlg_model_unit_length_x - 1
                dst_p1_x = wlg_model_unit_length_x - 1 - wlg_model_unit_length_x / 4
                dst_p1_y = 0
                src_plist.append(src_p[1])
                dst_plist.append([dst_p1_x, dst_p1_y])

                line_shape = linelist[-1]
                src_p = sorted(line_shape['points'], key = lambda k : k[0])

                #dst_p0_x = 0
                dst_p0_x = wlg_model_unit_length_x / 4
                dst_p0_y = (highlist_locH - float(line_shape['label'])) * wlg_model_unit_length_y - 1
                src_plist.append(src_p[0])
                dst_plist.append([dst_p0_x, dst_p0_y])

                dst_p1_x = wlg_model_unit_length_x - 1 - wlg_model_unit_length_x / 4
                #dst_p1_x = wlg_model_unit_length_x - 1
                dst_p1_y = (highlist_locH - float(line_shape['label'])) * wlg_model_unit_length_y - 1
                src_plist.append(src_p[1])
                dst_plist.append([dst_p1_x, dst_p1_y])


                src_p_array = np.array(src_plist, np.float32)
                dst_p_array = np.array(dst_plist, np.float32)

                #计算投影矩阵

                H_mitrix = cv2.getPerspectiveTransform(src_p_array, dst_p_array)
                #生成变换图像
                out_img = cv2.warpPerspective(img, H_mitrix, (wlg_model_unit_length_x, wlg_model_total_length), flags = cv2.INTER_CUBIC, borderMode =cv2.BORDER_REPLICATE)
                or_out_img = out_img

                #计算linestrip变换后的结果
                wl_s = []
                for wl_s_point in wl_strip[0]['points']:
                    wl_s_point.append(1)
                    wl_s_dst_point = np.dot(H_mitrix, wl_s_point)
                    wl_s_dst_point = [wl_s_dst_point[0]/ wl_s_dst_point[2], wl_s_dst_point[1]/ wl_s_dst_point[2]]
                    wl_s.append(wl_s_dst_point)
                wl_s.sort(key = lambda x:x[0])

                wl_s_dst = []
                
                for i in range(wlg_model_unit_length_x):
                    wl_s_dst_elm = 0
                    if i <= wl_s[0][0]:
                        wl_s_dst_elm = wl_s[0][1]
                    elif i >= wl_s[-1][0]:
                        wl_s_dst_elm = wl_s[-1][1]
                    else:
                        for j, val in enumerate(wl_s):
                            val_next = wl_s[j + 1]
                            if i > val[0] and i < val_next[0]:
                                w1 = (i - val[0]) / (val_next[0] - val[0])
                                wl_s_dst_elm = (1-w1) * val[1] + w1 * val_next[1]
                                break
                    if wl_s_dst_elm == 0:
                        print('!!!!!!!!!!!!!!!!!!!')
                        sys.exit(0)
                    wl_s_dst_elm = min(wlg_model_total_length, max(0, wl_s_dst_elm))
                    wl_s_dst.append(wl_s_dst_elm)  

                # 2.0版本
                wl_dst_point =[wlg_model_unit_length_x // 2,0]
                wl_dst_point[1] =  get_median(wl_s_dst[int(wlg_model_unit_length_x / 4 +0.5 ) :int(3 * wlg_model_unit_length_x / 4 + 0.5)])
                #wl_dst_point[1] =  get_median(wl_s_dst)
                or_wl_s_dst = wl_s_dst



                #生成模版图中的标注json
                out_image_name = image_name + str(int(shape['points'][0][0]))+'.png'
                out_json_name = image_name + str(int(shape['points'][0][0]))+'.json'
                out_json['imagePath'] = out_image_name
                out_json['image_w'] = wlg_model_unit_length_x
                out_json['image_h'] = wlg_model_total_length
                out_json['waterLevelLocal'] = min(max(0, wl_dst_point[1]),    wlg_model_total_length)
                out_json['waterLevelLocal_f'] = wl_dst_point[1] / wlg_model_total_length
                out_json['waterLevelStrip'] = wl_s_dst
                out_json['waterLevelStrip_f'] = [x / wlg_model_total_length for x in wl_s_dst]
                out_json['model_unit_length'] = wlg_model_unit_length_y
                out_json['model_unit_length_f'] = wlg_model_unit_length_y / wlg_model_total_length
                out_json['highlistLevel'] = highlist_locH
                out_json['waterLevel'] = highlist_locH - wl_dst_point[1] / wlg_model_unit_length_y



                

                with open(out_path + out_json_name,'w',encoding='utf-8') as f:
                    json.dump(out_json, f,indent = 2, ensure_ascii=False)

        return out_fix_linelist  #fix_linelist  所在场景只能存在一根水尺

if __name__ == '__main__':

    jFilePath_list = {}
    jFilePath_list_C = {}
    '''
    解析批量解析lebelme 的 json
    '''

    jFilePath_list = {
        '../data/original_val/*.json' : '../data/mapping_val/',
        }

    list_len = len(jFilePath_list)
    print('jFilePath_list dic len : ', list_len)

    for key in jFilePath_list:
        print( key , ':', jFilePath_list[key])

        json_file_path = key
        out_path = jFilePath_list[key]
        json_file = glob.glob(json_file_path)
        for json_full_name in json_file:
            read_json(json_full_name, out_path)


