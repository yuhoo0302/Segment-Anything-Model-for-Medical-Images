# %% import packages
import numpy as np
import os
from glob import glob
import pandas as pd

join = os.path.join
from skimage import transform, io, segmentation
from tqdm import tqdm
import torch
from torchvision.transforms.functional import InterpolationMode
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse

# set up the parser
parser = argparse.ArgumentParser(description="preprocess grey and RGB images")

# add arguments to the parser
parser.add_argument(
    "-i",
    "--img_path",
    type=str,
    default=f"data/test_data",
    help="path to the images",
)
parser.add_argument(
    "-gt",
    "--gt_path",
    type=str,
    default=f"data/test_data",
    help="path to the ground truth (gt)",
)

parser.add_argument(
    "-task",
    "--task_name",
    type=str,
    default=f"22_Heart",
    help="name to test dataset",
)

parser.add_argument(
    "--csv",
    type=str,
    default=None,
    help="path to the csv file",
)

parser.add_argument(
    "-o",
    "--npz_path",
    type=str,
    default=f"data",
    help="path to save the npz files",
)
parser.add_argument(
    "--data_name",
    type=str,
    default="demo2d",
    help="dataset name; used to name the final npz file, e.g., demo2d.npz",
)
parser.add_argument("--image_size", type=int, default=1024, help="image size")
parser.add_argument(
    "--img_name_suffix", type=str, default=".png", help="image name suffix"
)
# parser.add_argument("--label_id", type=int, default=255, help="label id")
parser.add_argument("--model_type", type=str, default="vit_b", help="model type")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="sam_vit_b_01ec64.pth",
    help="original sam checkpoint",
)
parser.add_argument("--device", type=str, default="cuda:1", help="device")
parser.add_argument("--seed", type=int, default=2023, help="random seed")

# parse the arguments
args = parser.parse_args()

# create a directory to save the npz files
save_base = args.npz_path + "/precompute_" + args.model_type

# convert 2d grey or rgb images to npz file
imgs = []
gts = []
img_embeddings = []

# set up the model
# get the model from sam_model_registry using the model_type argument
# and load it with checkpoint argument
# download save the SAM checkpoint.
# [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth](VIT-B SAM model)

sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(
    args.device
)

# ResizeLongestSide (1024), including image and gt
sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)

def process(gt_name: str, image_name: str, mode: str):
    if image_name == None:
        image_name = gt_name.split(".")[0] + args.img_name_suffix
    if mode == "train":
        gt_data = io.imread(join(args.gt_path, gt_name)) # H, W
    elif mode == "valid":
        gt_data = io.imread(join(args.gt_path.replace("train", "valid"), gt_name))
    else:
        gt_path = f"data/test_data/{args.task_name}/labels"
        gt_data = io.imread(join(gt_path, gt_name))
    # if it is rgb, select the first channel
    if len(gt_data.shape) == 3:
        gt_data = gt_data[:, :, 0]
    assert len(gt_data.shape) == 2, "ground truth should be 2D"

    # resize ground truth image
    # resize_gt = sam_transform.apply_image(gt_data, interpolation=InterpolationMode.NEAREST) # ResizeLong (resized_h, 1024)
    # gt_data = sam_model.preprocess_for_gt(resize_gt)

    # exclude tiny objects (considering multi-object)
    gt = gt_data.copy()
    label_list = np.unique(gt_data)[1:]
    del_lab = [] # for check
    for label in label_list:
        gt_single = (gt_data == label) + 0
        if np.sum(gt_single) <= 50:
            gt[gt == label] = 0
            del_lab.append(label)
    assert len(list(np.unique(gt)) + del_lab) == len(list(label_list) + [0])

    if np.sum(gt) > 0: # has at least 1 object
        # gt: seperate each target into size (B, H, W) binary 0-1 uint8
        new_lab_list = list(np.unique(gt))[1:] # except bk
        new_lab_list.sort()
        gt_ = []
        for l in new_lab_list:
            gt_.append((gt == l) + 0)
        gt_ = np.array(gt_, dtype=np.uint8)

        if mode == "train":
            image_data = io.imread(join(args.img_path, image_name))
        elif mode == "valid":
            image_data = io.imread(join(args.img_path.replace("train", "valid"), image_name))
        else:
            img_path = f"data/test_data/{args.task_name}/images"
            image_data = io.imread(join(img_path, image_name))
        image_ori_size = image_data.shape[:2]
        # Remove any alpha channel if present.
        if image_data.shape[-1] > 3 and len(image_data.shape) == 3:
            image_data = image_data[:, :, :3]
        # If image is grayscale, then repeat the last channel to convert to rgb
        if len(image_data.shape) == 2:
            image_data = np.repeat(image_data[:, :, None], 3, axis=-1)
        # nii preprocess start (clip the intensity)
        lower_bound, upper_bound = np.percentile(image_data, 0.95), np.percentile(
            image_data, 99.5 # Intensity of 0.95% pixels in image_data lower than lower_bound
                             # Intensity of 99.5% pixels in image_data lower than upper_bound
        )
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        # min-max normalize and scale
        image_data_pre = (
            (image_data_pre - np.min(image_data_pre))
            / (np.max(image_data_pre) - np.min(image_data_pre))
            * 255.0
        )
        image_data_pre[image_data == 0] = 0 # ensure 0-255
        image_data_pre = np.uint8(image_data_pre)
        imgs.append(image_data_pre)

        # resize image to 3*1024*1024
        resize_img = sam_transform.apply_image(image_data_pre, interpolation=InterpolationMode.BILINEAR) # ResizeLong
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1))[None, :, :, :].to(
            args.device
        ) # (1, 3, resized_h, 1024)
        resized_size_be_padding = tuple(resize_img_tensor.shape[-2:])
        input_image = sam_model.preprocess(resize_img_tensor) # padding to (1, 3, 1024, 1024)
        assert input_image.shape == (
            1,
            3,
            sam_model.image_encoder.img_size,
            sam_model.image_encoder.img_size,
        ), "input image should be resized to 1024*1024"
        assert input_image.shape[-2:] == (1024, 1024)
        # pre-compute the image embedding
        if mode != "train":
            sam_model.eval()
        with torch.no_grad():
            embedding = sam_model.image_encoder(input_image)
            img_embedding = embedding.cpu().numpy()[0]
        return gt_, new_lab_list, img_embedding, resized_size_be_padding, image_ori_size
    else:
        print(mode, gt_name)
        return None, None, None, None, None

if __name__ == "__main__":
    mode = 'test'
    if args.csv != None:
        # if data is presented in csv format
        # columns must be named image_filename and mask_filename respectively
        try:
            os.path.exists(args.csv)
        except FileNotFoundError as e:
            print(f"File {args.csv} not found!!")

        df = pd.read_csv(args.csv)
        bar = tqdm(df.iterrows(), total=len(df))
        for idx, row in bar:
            process(row.mask_filename, row.image_filename)

    else:
        # get all the names of the images in the ground truth folder
        if mode == 'train' or mode == 'valid':
            names = sorted(os.listdir(args.gt_path))
            # save
            save_path = join(save_base, mode)
        else:
            gt_path = f"{args.gt_path}/{args.task_name}/labels"
            args.img_path = f"{args.gt_path}/{args.task_name}/images"
            names = sorted(os.listdir(gt_path))
            # save
            save_path = join(save_base, mode,args.task_name)
        # print the number of images found in the ground truth folder
        print("Num. of all train images:", len(names))
        
        os.makedirs(save_path, exist_ok=True)
        for gt_name in tqdm(names):
            if os.path.exists(join(save_path, gt_name.split('.')[0] + ".npz")):
                continue
            img_name = gt_name.replace('_mask','')
            image_path = os.path.join(args.img_path, img_name)
            if not os.path.exists(image_path):
                continue
            gt_, new_lab_list, img_embedding, resized_size_be_padding, image_ori_size = process(gt_name, img_name, mode=mode)
            if gt_ is not None:
                np.savez_compressed(
                    join(save_path, gt_name.split('.')[0] + ".npz"),
                    label_except_bk=new_lab_list,
                    gts=gt_,
                    img_embeddings=img_embedding,
                    image_shape=image_ori_size,
                    resized_size_before_padding=resized_size_be_padding
                )
        print("Num. of processed train images (delete images with no any targets):", len(imgs))
