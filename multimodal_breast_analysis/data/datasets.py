# set of functions to return list of dicts for every dataset
import os
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
import torch

def penn_fudan(root):
    imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
    masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
    data = []
    for image, mask in zip(imgs, masks):
            img_path = os.path.join(root, "PNGImages", image)
            mask_path = os.path.join(root, "PedMasks", mask)
            mask = read_image(mask_path)
            obj_ids = torch.unique(mask)
            obj_ids = obj_ids[1:]
            num_objs = len(obj_ids)
            masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
            boxes = masks_to_boxes(masks)
            labels = torch.ones((num_objs,), dtype=torch.int64)
            target = {}
            target["image"] = img_path
            target["boxes"] = boxes
            target["labels"] = labels
            data.append(target)
    return data