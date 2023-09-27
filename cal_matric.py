import numpy as np
import cv2
from collections import Counter
from get_score import *
import glob
from tqdm import tqdm
import json
import os
import time
import pandas as pd
from collections import Counter

# ['DICE', 'JAC', 'HD', 'ASD', 'CON']
def calcute_value(result, reference): 
    dc_val= dc(result, reference)
    d2c_val = d2c(dc_val)
    jc_val = jc(result, reference)
    hd_val = hd(result, reference)
    asd_val = asd(result, reference)
    
    d2c_val = d2c_val *100
    dc_val = dc_val * 100
    jc_val = jc_val * 100
    return [dc_val, jc_val, hd_val, asd_val, d2c_val]

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
    y_true_f = y_true.reshape(y_true.shape[0], -1)
    y_pred_f = y_pred.reshape(y_pred.shape[0], -1)
    intersection = np.matmul(y_true_f.astype(float), y_pred_f.T.astype(float))
    dice_matrix = (2. * intersection + smooth) / (y_true_f.sum(1).reshape(y_true_f.shape[0],-1) + y_pred_f.sum(1) + smooth)
    dice_max_index, dice_max_value = dice_matrix.argmax(1), dice_matrix.max(1)
    return dice_matrix, dice_max_index, dice_max_value


if __name__ == "__main__":
    # task_name
    task = "22_Heart"
    task_name = task.split('_')[1]
    test_mode = "finetune_data"
    print("Current processing task: ", task)

    # model size
    size = "b" 
    
    # save results
    p = f"work_dir_{size}/{test_mode}/indicator_results/{task}"
    os.makedirs(p, exist_ok=True)
    
    mode = "box"
    max_label_list = []
    all_dices = {}
    all_dices[mode] = {}
    all_imgs_p = {}
    all_imgs_p[mode] = {}
    
    cal_method = ['DICE', 'JAC', 'HD', 'ASD', 'CON']
    info = json.load(open(f"data_infor_json/{task}.json"))
    label_info = info["info"]
    color_info = info["color"]
    
    
    with tqdm(total=len(glob.glob(f"data/test_data/{task}/images/*")), desc=f'Current mode: {mode}', mininterval=0.3) as pbar:
        value_targets = [[] for _ in range(len(label_info))]
        img_p_targets = [[] for _ in range(len(label_info))] 
        # print(len(value_targets))
        image_index = 0

        for ori_path in glob.glob(f"data/test_data/{task}/images/*"):
            pbar.update(1)

            name = ori_path.split('/')[-1].split('.')[0]
            if os.path.exists(os.path.join(f"data/precompute_vit_{size}/test", task, name + ".npz")):
                npz_data = np.load(os.path.join(f"data/precompute_vit_{size}/test", task, name + ".npz"))
            else:
                continue
            label_list = npz_data['label_except_bk'].tolist()
            gt2D = npz_data['gts']

            if os.path.exists(f"work_dir_{size}/{test_mode}/{task}/pred_only_box/{name}.npy"):
                pred_files = np.load(f"work_dir_{size}/{test_mode}/{task}/pred_only_box/{name}.npy")
            else:
                continue
            all_info = {}
            
            for idx in range(pred_files.shape[0]):
                current_label = label_list[idx]
                current_gt = gt2D[idx]
                dice_matrix, dice_max_index, _ = dice_coefficient(current_gt[None, :, :]*1, pred_files[idx]*1)
                matrix = calcute_value(pred_files[idx][dice_max_index.squeeze(0)], current_gt) # ['DICE', 'JAC', 'HD', 'ASD', 'CON']
                id_dice = int(list(color_info.keys())[list(color_info.values()).index(label_list[idx])])
                value_targets[id_dice - 1].append(matrix)
                img_p_targets[id_dice - 1].append(name)
        
        all_dices[mode] = value_targets
        all_imgs_p[mode] = img_p_targets

    print("indiactor: ", ['DICE', 'JAC', 'HD', 'ASD', 'CON'])
    
    # 1. record all the indicators (mean(std)) into csv
    # 2. record all the indicators (per image) of different targets (labels) and modes into csv
    sub_headers = "Img_name,DICE,JAC,HD,ASD,CON"
    headers = " ,box, , , , ,"
    all_mode_matrix = []
    all_label_matrix = []
            
    all_obj_matrix = []
    label_len = len(label_info)
    for obj_i in range(len(label_info)):
        
        if len(all_label_matrix) < label_len:
            all_label_matrix.append(np.array([label_info[str(obj_i + 1)]]))
            
        current_matrix = np.array(all_dices[mode][obj_i])
        current_imgs_p = np.array(all_imgs_p[mode][obj_i])[:, None]
        current_info = np.concatenate((current_imgs_p, current_matrix), axis=1)
        
        np.savetxt(f"{p}/{task}_{mode}_{label_info[str(obj_i + 1)]}.csv", current_info, delimiter=",", fmt = "%s", header=sub_headers, comments="")
        print(f"finish recording {mode}_{label_info[str(obj_i + 1)]} results of dataset {task}")
        
        mean_matrix = np.mean(current_matrix, axis=0)
        std_matrix = np.std(current_matrix, axis=0)
        all_obj_matrix.append(np.array([f"{round(mean_matrix[x], 2)}({round(std_matrix[x], 2)})" for x in range(mean_matrix.shape[0])]))
    all_mode_matrix.append(np.stack(all_obj_matrix, 0))
    all_mode_matrix = np.hstack(all_mode_matrix)
    all_label_matrix = np.vstack(all_label_matrix)
    assert all_mode_matrix.shape == (label_len, 5 * 1)

    np.savetxt(f"{p}/{task}_mean(std).csv", np.hstack((all_label_matrix, all_mode_matrix)), delimiter=",", fmt = "%s", header=headers, comments="")
    print(f"finish recording mean(std) results of dataset {task}")

    
