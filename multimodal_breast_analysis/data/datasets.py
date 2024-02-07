# set of functions to return list of dicts for every dataset
import os
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
import torch
import pandas as pd
import numpy as np

def penn_fudan(root):
    imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
    masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
    data = []
    groups = []
    for idx, (image, mask) in enumerate(zip(imgs, masks)):
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
            groups.append(idx)
    return data, groups


def omidb(csv_path):
    df = pd.read_csv(csv_path)
    df = df.loc[df["manufacturer"] == 'HOLOGIC, Inc.']
    df = df.loc[df["num_marks"] == 1]
    # df = df.loc[df["view_position"] == 'CC']

    dataset = []
    groups = []
    for i in range(len(df)):
        row = df.iloc[i]
        image_path = row["path"]
        boxes = np.asarray([row["bbox"].replace('(', '').replace(')', '').split(',')], dtype = int)
        if boxes.min() < 0:
            continue
        labels = np.asarray([1])
        dataset.append({"image": image_path, "boxes": boxes, "labels": labels})
        groups.append([row['client']])
    return dataset, groups



def omidb_downsampled(img_dir):
    csv_file = os.path.join(img_dir, "omidb-selection.csv")
    df = pd.read_csv(csv_file)
    df = df.loc[df["scanner"] == 'HOLOGIC']
    dataset = []    
    groups = []
    for idx, row in df.iterrows():
        if idx in [552, 1142, 3008, 3461, 3645, 3682]: # negative coordinates/roi larger than image
            continue
        filename = os.path.join(img_dir+"/HOLOGIC/ffdm/st"+"{0:03}".format(row["subtype"]), row["filename"])
        bbox = row["bbox"][12:-1]
        x_crop,y_crop,_,_ = [int(value.split('=')[1]) for value in bbox.split(', ')]
        bbox_roi = row["bbox_roi"][12:-1]
        xmin, ymin, xmax, ymax = [int(value.split('=')[1]) for value in bbox_roi.split(', ')]
        xmin, ymin, xmax, ymax = xmin - x_crop, ymin - y_crop, xmax - x_crop, ymax - y_crop
        dataset.append({"image": filename, "boxes": np.asarray([[xmin,ymin,xmax,ymax]], dtype = int), "labels": np.asarray([1]),
                        "client": row['client'], "filename": filename})
        groups.append(row['client'])
    return dataset, groups

    
def dbt_2d(data_path, metadata_path):
    metadata = pd.read_csv(metadata_path)
    # metadata = metadata.drop_duplicates(subset=['PatientID', 'StudyUID', 'View'], keep=False) #single lesion only
    dataset = []    
    groups = []
    for path in os.listdir(data_path):
        patient_id, study_id, view, slice = path.split('.')[0].split('_')
        image_path = data_path + path
        rows = metadata[(metadata['PatientID'] == patient_id) & (metadata['StudyUID'] == study_id) & (metadata['View'] == view)]
        rows = rows[(float(slice) >= rows['Slice'] - 0.25 * rows['VolumeSlices']) & (float(slice) <= rows['Slice'] + 0.25 * rows['VolumeSlices'])]
        boxes = rows[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(int)
        labels = np.ones(boxes.shape[0])
        central_slice = rows['Slice'].values
        if int(slice) in central_slice:
            dataset.append({"image": image_path, "boxes": boxes, "labels": labels,
                            "PatientID": patient_id, "StudyUID": study_id, "View": view, "Slice":slice, "CentralSlice": central_slice})
            groups.append(patient_id)
    return dataset, groups

