# %% set up environment
import numpy as np
import matplotlib.pyplot as plt
import os
import random
join = os.path.join
from tqdm import tqdm
from time import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils.get_prompts import find_all_info
import monai
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse

# set seeds
torch.manual_seed(3407)
np.random.seed(2023)

def collate_fn(batch):
    return batch

#%% create a dataset class to load npz data and return back image embeddings and ground truth
class NpzDataset(Dataset): 
    def __init__(self, data_root):
        self.data_root = data_root
        self.npz_files = sorted(os.listdir(self.data_root)) 
        self.resize_transform = ResizeLongestSide(target_length=1024)
    
    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, index):
        npz_data = np.load(join(self.data_root, self.npz_files[index]))
        gt2D = npz_data['gts']
        label_list = npz_data['label_except_bk'].tolist()
        img_embed = npz_data['img_embeddings']
        _, _, box_list = find_all_info(mask=gt2D, 
                                       label_list=label_list, 
                                       point_num=10
                                       )
        # convert img embedding, gt_mask, bounding box to torch tensor
        box = torch.tensor(box_list).float() # B, 4
        img_embed = torch.tensor(img_embed).float()
        gt2D = torch.tensor(gt2D).long() # B, ori_H, ori_W

        # scale to original size
        box = self.resize_transform.apply_boxes_torch(box, (gt2D.shape[-2], gt2D.shape[-1]))
        
        return {"img_embed": img_embed,
                "gt2D": gt2D,
                "box": box,
                "image_ori_size": npz_data['image_shape'],
                "size_before_pad": npz_data['resized_size_before_padding']
                }
        
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

def train_sam(model: nn.Module, optimizer, train_loader, epoch, device, criterion):
    epoch_start_time = time()
    epoch_loss = 0
    model.train()
    print("==========Training==========")
    for step, batched_input in enumerate(tqdm(train_loader)):
        outputs = []
        # do not compute gradients for image encoder and prompt encoder
        with torch.no_grad():
            none_grad_features = {"sparse": {}, "dense": {}}
            for idx, image_record in enumerate(batched_input):
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None,
                    boxes=image_record["box"].to(device),
                    masks=None,
                )
                none_grad_features["sparse"][idx] = sparse_embeddings
                none_grad_features["dense"][idx] = dense_embeddings

        batched_loss = 0
        for id, im_record in enumerate(batched_input):
            # low_res_masks.shape == (B, M, 256, 256) M is set to 1
            low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=im_record["img_embed"].unsqueeze(0).to(device), # (1, 256, 64, 64) !!1 = batch size
                image_pe=model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64) !!1 = batch size
                sparse_prompt_embeddings=none_grad_features["sparse"][id], # (B, 2, 256) !!B = target num instead of batch size
                dense_prompt_embeddings=none_grad_features["dense"][id], # (B, 256, 64, 64) !!B = target num instead of batch size
                multimask_output=False,
            )
            masks = model.postprocess_masks( # upscale + eliminate padding + restore to ori size
                low_res_masks,
                input_size=tuple(im_record["size_before_pad"]),
                original_size=tuple(im_record["image_ori_size"]),
            )
            outputs.append({
                "masks": masks,
                "iou_predictions": iou_predictions,
                "low_res_logits": low_res_masks,
                "gt2D": im_record["gt2D"].to(device)
            })
            # first ele: 1, B, ori_H, ori_W
            # second ele: 1, B, ori_H, ori_W
            batched_loss += criterion(masks.squeeze(1).unsqueeze(0), im_record["gt2D"].to(device).unsqueeze(0)) # considering the multi-object situation

        loss = batched_loss / len(batched_input)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= (step + 1)
    print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}')
    return epoch_loss, time() - epoch_start_time

def valid_sam(model: nn.Module, valid_loader, epoch, device):
    model.eval()
    epoch_dsc = 0.
    print("==========Validation==========")
    for step, batched_input in enumerate(tqdm(valid_loader)):
        outputs = []
        with torch.no_grad():
            batched_dsc = 0.
            for image_record in batched_input:
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None,
                    boxes=image_record["box"].to(device),
                    masks=None,
                )
                # low_res_masks.shape == (B, M, 256, 256) M is set to 1
                low_res_masks, iou_predictions = model.mask_decoder(
                    image_embeddings=image_record["img_embed"].unsqueeze(0).to(device), # (1, 256, 64, 64) !!1 = batch size
                    image_pe=model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64) !!1 = batch size
                    sparse_prompt_embeddings=sparse_embeddings, # (B, N, 256) !!B = target num instead of batch size
                    dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64) !!B = target num instead of batch size
                    multimask_output=True,
                )
                masks = model.postprocess_masks( # upscale + eliminate padding + restore to ori size
                    low_res_masks,
                    input_size=tuple(image_record["size_before_pad"]),
                    original_size=tuple(image_record["image_ori_size"]),
                )
                outputs.append({
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                    "gt2D": image_record["gt2D"].to(device)
                })
                # compute foreground dice
                masks = masks > model.mask_threshold
                masks = masks.cpu().numpy() # B, 1, ori_H, ori_W
                gt2D = np.array(image_record["gt2D"]) # B, ori_H, ori_W
                target_mean_dsc = 0.
                for i in range(len(gt2D)): # each target
                    cur_gt = gt2D[i]
                    _, _, dice_max_value = dice_coefficient(y_true=cur_gt[None, :, :], y_pred=masks[i])
                    target_mean_dsc += dice_max_value.squeeze(0)
                batched_dsc += target_mean_dsc / len(gt2D)
        
        dsc = batched_dsc / len(batched_input)
        epoch_dsc += dsc
    epoch_dsc /= (step + 1)
    print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, DSC: {epoch_dsc}')
    return epoch_dsc

if __name__ == "__main__":
    # Task = "Heart_2d"
    # %% set up parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--tr_npz_path', type=str, default=f'data/precompute_vit_b/train', help='path to training npz files (im_emb, gt)')
    parser.add_argument('-j', '--val_npz_path', type=str, default=f'data/precompute_vit_b/train', help='path to validation npz files (im_emb, gt)')
    parser.add_argument('--model_type', type=str, default='vit_b')
    parser.add_argument('--checkpoint', type=str, default='sam_vit_b_01ec64.pth', help='original sam checkpoint path')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--work_dir', type=str, default='work_dir_b')
    # train
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    args = parser.parse_args()


    # %% set up model for fine-tuning 
    Task = "finetune_data"
    device = args.device
    model_save_path = join(args.work_dir, Task)
    os.makedirs(model_save_path, exist_ok=True)
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint, device=device).to(device)

    # Set up the optimizer, hyperparameter tuning will improve performance here
    optimizer = torch.optim.AdamW(sam_model.mask_decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    #%% train & valid
    num_epochs = args.num_epochs
    losses = []
    dscs = []
    times = []
    best_loss, best_dsc = 1e10, 0.
    train_dataset = NpzDataset(args.tr_npz_path)
    val_dataset = NpzDataset(args.val_npz_path)
    print("Number of training samples: ", len(train_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                                  pin_memory=True, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                                  pin_memory=True, shuffle=False, collate_fn=collate_fn)
    for epoch in range(num_epochs):
        # train
        epoch_loss, runtime = train_sam(sam_model, optimizer, train_dataloader, epoch, device, criterion=seg_loss)
        losses.append(epoch_loss)
        times.append(runtime)
        # valid
        epoch_dsc = valid_sam(sam_model, valid_dataloader, epoch, device)
        dscs.append(epoch_dsc)

        # save the model checkpoint
        torch.save(sam_model.state_dict(), join(model_save_path, f'medsam_box_last.pth'))
        # save the best model
        if epoch_dsc > best_dsc:
            best_dsc = epoch_dsc
            print("Update: saving {} model as the best checkpoint".format(epoch))
            torch.save(sam_model.state_dict(), join(model_save_path, f'medsam_box_best.pth'))

        # %% plot loss
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,4))
        ax1.plot(losses)
        ax1.set_title("Dice + Cross Entropy Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax2.plot(dscs)
        ax2.set_title("Dice of valid set")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Dice")
        ax3.plot(times)
        ax3.set_title("Epoch Running Time")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Time (s)")
        fig.savefig(join(model_save_path, f"medsam_box_record.png"))
        plt.close()

