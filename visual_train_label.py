from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import cv2
import os
from copy import deepcopy
import glob
from tqdm import tqdm
from segment_anything.utils.transforms import ResizeLongestSide
import torch
from collections import Counter
import json
from PIL import Image

join = os.path.join
    
def show_points(image, coords, labels):
    ret_img = deepcopy(image)
    if labels == 1:
        cv2.circle(ret_img, (coords[0][0],coords[0][1]), radius=5,  color=(0,255,0), thickness=-1)
    else:
        cv2.circle(ret_img, (coords[0][0],coords[0][1]), radius=5, color=(0,0,255),  thickness=-1)
    return ret_img

def dice_coefficient(y_true, y_pred):
    smooth = 1
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def limit_rect(mask, box_ratio):
    height, width = mask.shape[0], mask.shape[1]
    box = find_box_from_mask(mask)
    w, h = box[2] - box[0], box[3] - box[1]
    w_ratio = w * box_ratio
    h_ratio = h * box_ratio
    x1 = box[0] - w_ratio/2 + w / 2
    y1 = box[1] - h_ratio/2 + h / 2
    x2 = x1 + w_ratio
    y2 = y1 + h_ratio
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 >= width:
        x2 = width
    if y2 >= height:
        y2 = height
    return x1, y1, x2-x1, y2-y1

def find_center_from_mask(mask):
    # calculate moments of binary image
    M = cv2.moments(mask)
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return [cX, cY], 1

def find_center_from_mask_new(mask, box_ratio=2, n_fg=5, n_bg=5):
# def get_all_point_info(mask, box_ratio, n_fg, n_bg):
    """
    input:
        mask:     single mask
        bg_ratio: Expanded bg_ratio times based on the largest external frame
        n_fg:     Number of foreground points
        n_bg:     Number of background points
    Return:
        point_coords(ndarry): size=M*2, select M point (foreground or background)
        point_labels(ndarry): size=M 
    """
    # find center of mass
    M = cv2.moments(mask)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    center_point = np.array([cX, cY]).reshape(1, 2)

    # acquire foreground points
    indices_fg = np.where(mask == 1)
    points_fg = np.column_stack((indices_fg[1], indices_fg[0]))

    # uniformly sample n points
    step_fg = int(len(points_fg) / n_fg)
    # print(len(points_fg))
    points_fg = points_fg[::step_fg, :]
    

    # find the maximum bounding box
    x, y, w, h = limit_rect(mask, box_ratio)
    box1 = (x, y, x+w, y+h)
    x, y, w, h = int(x), int(y), int(w), int(h)

    # acquire background points
    yy, xx = np.meshgrid(np.arange(x, x+w), np.arange(y, y+h))
    roi = mask[y:y+h, x:x+w]
    bool_mask = roi == 0
    points_bg = np.column_stack((yy[bool_mask], xx[bool_mask]))

    # uniformly sample n points
    step_bg = int(len(points_bg) / n_bg)
    points_bg = points_bg[::step_bg, :]

    # get point_coords
    points_fg = np.concatenate((center_point, points_fg[1:]), axis=0)
    point_coords = np.concatenate((points_fg, points_bg), axis=0)
    point_labels = np.concatenate((np.ones(n_fg), np.zeros(n_bg)), axis=0)

    return point_coords, point_labels, points_fg, points_bg, box1, (cX, cY) 

   # find center of mass
def find_box_from_mask(mask):
    y, x = np.where(mask == 1)
    x0 = x.min()
    x1 = x.max()
    y0 = y.min()
    y1 = y.max()
    return [x0, y0, x1, y1]

    # acquire information about points and box
def find_all_info(mask, label_list):
    point_list = []
    point_label_list = []
    mask_list = []
    box_list = []
    # multi-object processing
    for current_label_id in range(len(label_list)):
        current_mask = mask[current_label_id]
        current_center_point_list, current_label_list,_,_,_,_=  find_center_from_mask_new(current_mask)
        current_box = find_box_from_mask(current_mask)
        point_list.append(current_center_point_list[0:10,:])
        point_label_list.append(current_label_list[0:10,])
        mask_list.append(current_mask)
        box_list.append(current_box)
    return point_list, point_label_list, box_list, mask_list

def read_image_mask(image_path, mask_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt = cv2.imread(mask_path,0)
    return image, gt

def get_one_target_final_dice(masks, gt):
    dice_list = []
    for i in range(len(masks)):
        dice_list.append(dice_coefficient(masks[i], gt))
    # choose the largest dice (GT <-> mask)
    res_mask = masks[np.argmax(dice_list)]
    return dice_list[np.argmax(dice_list)], res_mask

def dice_coefficient(y_true, y_pred):
    """
    y_true: GT, [N, W, H]
    Y_pred: target, [M, W, H]
    N, M: number
    W, H: weight and height of the masks
    Returns:
        dice_matrix [N, M]
        dice_max_index [N,] indexes of prediceted masks with the highest DICE between each N GTs 
        dice_max_value [N,] N values of the highest DICE
    """
    smooth = 0.1


    y_true_f = y_true.reshape(y_true.shape[0], -1) # N
    
    y_pred_f = y_pred.reshape(y_pred.shape[0], -1) # M
    intersection = np.matmul(y_true_f.astype(float), y_pred_f.T.astype(float))
    dice_matrix = (2. * intersection + smooth) / (y_true_f.sum(1).reshape(y_true_f.shape[0],-1) + y_pred_f.sum(1) + smooth)
    dice_max_index, dice_max_value = dice_matrix.argmax(1), dice_matrix.max(1)
    return dice_matrix, dice_max_index, dice_max_value

def test_single_dataset(task, dataset_list, size):
    print("Current processing task: ", task)
    
    info = json.load(open(f"data_all/{task}/{task.split('_')[1]}_2d/{task}.json"))
    label_info = info["info"]
    color_info = info["color"]
    labels = []

    # dice结果
    with tqdm(total=len(dataset_list), desc=f'Current mode: box', mininterval=0.3) as pbar:
        for single_file in dataset_list:
            name = single_file.split('.')[0]
            ori_path = os.path.join("/mnt/data1/liulian/SAM_liulian/fine_tune/data/train_resize/images", single_file)

            if os.path.exists(join(f"/mnt/data1/liulian/SAM_liulian/fine_tune/data/precompute_vit_{size}/train", name + ".npz")):
                npz_data = np.load(join(f"/mnt/data1/liulian/SAM_liulian/fine_tune/data/precompute_vit_{size}/train", name + ".npz"))
            else:
                continue
            label_list = npz_data['label_except_bk'].tolist()
            for single_label in label_list:
                for single_c in color_info.keys():
                    if single_label == color_info[single_c]:
                        single = label_info[single_c]
                        if single not in labels:
                            labels.append(single)
                        else:
                            if len(labels) == len(color_info.keys()):
                                return labels  
    return labels
           
        
def process_valid(valid_path):
    file_list = os.listdir(valid_path)
    record_dict = {}
    for single_file in file_list:
        name_list = single_file.split('_')
        dataset_name = f"{name_list[0]}_{name_list[1]}"
        if dataset_name not in record_dict.keys():
            record_dict[dataset_name]=[]
        record_dict[dataset_name].append(single_file)
    return record_dict
    
if __name__ == "__main__":
    dataset_path = "/mnt/data1/liulian/SAM_liulian/fine_tune/data_all"
    valid_path = "/mnt/data1/liulian/SAM_liulian/fine_tune/data/train_resize/images"
    dataset_file_list = process_valid(valid_path)
    datset_list = os.listdir(dataset_path)
    record_dict = {}
    size = 'b'
    for single_dataset in dataset_file_list.keys():
        if "122_" in single_dataset:
            continue
        dataet_file_list = dataset_file_list[single_dataset]
        label_list = test_single_dataset(single_dataset, dataet_file_list, size)
        record_dict[single_dataset] = label_list
    
    save_path = f"/mnt/data1/liulian/SAM_liulian/fine_tune/train_label.json"
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(record_dict, f,  indent=4, ensure_ascii=False)
