# set of functions to return list of dicts for every dataset
import os
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
import torch
import pandas as pd

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


def omidb(img_dir):
    csv_file = os.path.join(img_dir, "omidb-selection.csv")
    df = pd.read_csv(csv_file)
    df_hologic = df.loc[df["scanner"] == 'HOLOGIC']
    df.head()
    dataset = []    
    failed_counter = 0
    for idx, row in df_hologic.iterrows():
        if idx in [
                    166, 242, 350, 387, 603, 985, 1008, 1142, 1330, 1336, 1353, 1359, 1393, 1412, 1592, 1664, 1677, 1684, 1765, 1766, 
                    1966, 2006, 2014, 2443, 2467, 2985, 3008, 3023, 3027, 3261, 3329, 3418, 3461, 3574, 3645, 3682, 3687, 3717
                ]:
            continue # corrupted coordinates
        filename = os.path.join(img_dir+"/HOLOGIC/ffdm/st"+"{0:03}".format(row["subtype"]), row["filename"])
        bbox_roi = row["bbox_roi"][12:-1]
        coords = bbox_roi.split(',')
        x1 = int(coords[0].split('=')[-1])
        y1 = int(coords[1].split('=')[-1])
        x2 = int(coords[2].split('=')[-1])
        y2 = int(coords[3].split('=')[-1])  
        data = {
                "image": filename, 
                "boxes": torch.tensor([[x1,y1,x2,y2]]),
                "labels": torch.tensor([1])
                }         
        dataset.append(data)
    print(len(dataset), failed_counter)
    return dataset