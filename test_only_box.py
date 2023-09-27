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

    # 计算dice
def dice_coefficient(y_true, y_pred):
    smooth = 1
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def limit_rect(mask, box_ratio):
    """ 判断扩大之后的目标框有没有超出图像 """
    height, width = mask.shape[0], mask.shape[1]
    # 最大外接框
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

    # 找质心
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
        mask:     单个目标的mask
        bg_ratio: 基于最大外接框的基础上扩张bg_ratio倍
        n_fg:     前景点个数
        n_bg:     背景点个数
    Return:
        point_coords(ndarry): size=M*2, 选取M个点(前景或背景)
        point_labels(ndarry): size=M, M个点的Label, 1为前景, 0为背景
    """
    # 找质心
    M = cv2.moments(mask)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    center_point = np.array([cX, cY]).reshape(1, 2)

    # 获得前景点
    indices_fg = np.where(mask == 1)
    points_fg = np.column_stack((indices_fg[1], indices_fg[0]))

    # 均匀采样n个点
    step_fg = int(len(points_fg) / n_fg)
    # print(len(points_fg))
    points_fg = points_fg[::step_fg, :]
    

    # 找最大外接框
    x, y, w, h = limit_rect(mask, box_ratio)
    box1 = (x, y, x+w, y+h)
    x, y, w, h = int(x), int(y), int(w), int(h)

    # 获得背景点
    yy, xx = np.meshgrid(np.arange(x, x+w), np.arange(y, y+h))
    roi = mask[y:y+h, x:x+w]
    bool_mask = roi == 0
    points_bg = np.column_stack((yy[bool_mask], xx[bool_mask]))

    # 均匀采样n个点
    step_bg = int(len(points_bg) / n_bg)
    points_bg = points_bg[::step_bg, :]

    # 获取point_coords
    points_fg = np.concatenate((center_point, points_fg[1:]), axis=0)
    point_coords = np.concatenate((points_fg, points_bg), axis=0)
    point_labels = np.concatenate((np.ones(n_fg), np.zeros(n_bg)), axis=0)

    return point_coords, point_labels, points_fg, points_bg, box1, (cX, cY) 

    # 找质心
def find_box_from_mask(mask):
    y, x = np.where(mask == 1)
    x0 = x.min()
    x1 = x.max()
    y0 = y.min()
    y1 = y.max()
    return [x0, y0, x1, y1]

    # 获取点和box的信息
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

if __name__ == "__main__":
    # task_name
    task = "22_Heart"
    print("Current processing task: ", task)
    
    # model size
    size = "b" 
        
    sam_checkpoint = f"work_dir_b/finetune_data/medsam_box_best.pth"
    json_info_path = f"data_infor_json/{task}.json"
    test_mode = "finetune_data"
    model_type = f"vit_{size}"
    device = "cuda:1"
        
    """construct model and predictor"""
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    sam.eval()
    predictor = SamPredictor(sam)
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

    #acquire dataset infomarion
    info = json.load(open(json_info_path))
    label_info = info["info"]
    color_info = info["color"]

    # dice结果
    all_dices = {}
    
    # save infer results (png: final mask, npy: pred_all_masks)
    # save selected prompts (npz)
    save_base = f"work_dir_{size}/{test_mode}/{task}/pred_only_box"
    save_prompt = f"work_dir_{size}/{test_mode}/{task}/prompts_only_box"
    os.makedirs(save_base, exist_ok=True)
    os.makedirs(save_prompt, exist_ok=True)
    
    all_dices["box"] = {}
    dice_targets = [[] for _ in range(len(label_info))] # 某个方法中,记录不同结构的dice
    
    with tqdm(total=len(glob.glob(f"data/test_data/{task}/images/*")), desc=f'Current mode: box', mininterval=0.3) as pbar:
        for ori_path in glob.glob(f"data/test_data/{task}/images/*"):
            name = ori_path.split('/')[-1].split('.')[0]

            if os.path.exists(join(f"data/precompute_vit_{size}/test", task, name + ".npz")):
                npz_data = np.load(join(f"data/precompute_vit_{size}/test", task, name + ".npz"))
            else:
                continue
            gt2D = npz_data['gts']
            label_list = npz_data['label_except_bk'].tolist()
            img_embed = torch.tensor(npz_data['img_embeddings']).float()

            # prompt mode
            predictor.original_size = tuple(npz_data['image_shape'])
            predictor.input_size = tuple(npz_data['resized_size_before_padding'])
            predictor.features = img_embed.unsqueeze(0).to(device)
            
            if not os.path.exists(os.path.join(save_base, ori_path.split('/')[-1].replace('png', 'npy'))):
                if not os.path.exists(f"{save_prompt}/{name}_prompts.npz"):#如果不存在就创建一个prompts.npz文件
                    _, _, box_list, gt_list = find_all_info(gt2D, label_list)
                    box_list = np.array(box_list)
                    np.savez(f"{save_prompt}/{name}_prompts.npz", box_list, gt_list)
                else:
                    info = np.load(f"{save_prompt}/{name}_prompts.npz")  
                    box_list, gt_list = info['arr_0'], info['arr_1']
                    
                # pre_process
                box_list_tensor = torch.tensor(box_list).float().to(device)
                box_list_tensor = resize_transform.apply_boxes_torch(box_list_tensor, (gt2D.shape[-2], gt2D.shape[-1]))
                    
                '''box'''
                masks, scores, logits = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes = box_list_tensor,
                    multimask_output=True,
                    ) # Mask -> N,M,H,W
                
                masks = masks.cpu().numpy()  
                np.save(os.path.join(save_base, ori_path.split('/')[-1].replace('png', 'npy')), masks)
                
            else:
                gt_list = np.load(f"{save_prompt}/{name}_prompts.npz")['arr_1']
                masks = np.load(os.path.join(save_base, ori_path.split('/')[-1].replace('png', 'npy')), allow_pickle=True)[()]
            
            # compute dice 
            current_method_res = np.zeros(gt2D.shape[1:]) # all target in a single image

            
            for idx in range(len(gt_list)): # mask list for a single image
                current_gt = gt_list[idx]
                dice_matrix, dice_max_index, dice_max_value = dice_coefficient(y_true=current_gt[None, :, :], y_pred=masks[idx])
                
                final_mask = masks[idx][dice_max_index.squeeze(0)]
                
                # index mapping for matching DICE with different target structures
                try:
                    id_dice = int(list(color_info.keys())[list(color_info.values()).index(label_list[idx])])
                    dice_targets[id_dice - 1].append(dice_max_value.squeeze(0))
                    
                    current_method_res[final_mask == 1] = label_list[idx] # one target
                except:
                    print(f"error: {ori_path}")
                    continue
            
            # for visulize (save infer results)
            if not os.path.exists(os.path.join(save_base, ori_path.split('/')[-1])):
                cv2.imwrite(os.path.join(save_base, ori_path.split('/')[-1]), current_method_res.astype(np.uint8))
                    
            pbar.update(1)
            
        # print
        for id in range(len(dice_targets)):
            all_dices["box"][label_info[str(id + 1)]] = f'{round(np.array(dice_targets[id]).mean() * 100, 2)}({round((100 * np.array(dice_targets[id])).std(), 2)})'
        
        print("======following is the dice results======")
        print(json.dumps(all_dices, indent=4, ensure_ascii=False))


        