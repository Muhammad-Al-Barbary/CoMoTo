"""
This module includes the necessary functions for reading the data. 
If the data paths are different the functions should be modified accordingly.
Each function should return a dataset list of dicts as well as a list of cases identifiers
"""
import os
import pandas as pd
import numpy as np


def omidb(img_dir):
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

    
def dbt(data_path, metadata_path, central_only = True):
    metadata = pd.read_csv(metadata_path)
    dataset = []    
    groups = []
    for path in os.listdir(data_path):
        patient_id, study_id, view, slice = path.split('.')[0].split('_')
        image_path = data_path + path
        rows = metadata[(metadata['PatientID'] == patient_id) & (metadata['StudyUID'] == study_id) & (metadata['View'] == view)]
        if central_only:
            rows = rows[float(slice) == rows['Slice']]
        else: # 25% around central slice according to challenge
            rows = rows[(float(slice) >= rows['Slice'] - 0.25 * rows['VolumeSlices']) & (float(slice) <= rows['Slice'] + 0.25 * rows['VolumeSlices'])]
        boxes = rows[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(int)
        labels = np.ones(boxes.shape[0])
        central_slice = rows['Slice'].values
        if len(boxes) != 0:
            dataset.append({"image": image_path, "boxes": boxes, "labels": labels,
                            "PatientID": patient_id, "StudyUID": study_id, "View": view, "Slice":slice, "CentralSlice": central_slice})
            groups.append(patient_id)
    #TODO: Fix this hardcoded merge of training and validation sets
    data_path = "../datasets/dbt/valid/2d/"
    metadata = pd.read_csv("../datasets/dbt/metadata_valid.csv")
    for path in os.listdir(data_path):
        patient_id, study_id, view, slice = path.split('.')[0].split('_')
        image_path = data_path + path
        rows = metadata[(metadata['PatientID'] == patient_id) & (metadata['StudyUID'] == study_id) & (metadata['View'] == view)]
        if central_only:
            rows = rows[float(slice) == rows['Slice']]
        else:
            rows = rows[(float(slice) >= rows['Slice'] - 0.25 * rows['VolumeSlices']) & (float(slice) <= rows['Slice'] + 0.25 * rows['VolumeSlices'])]
        boxes = rows[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(int)
        labels = np.ones(boxes.shape[0])
        central_slice = rows['Slice'].values
        if len(boxes) != 0:
            dataset.append({"image": image_path, "boxes": boxes, "labels": labels,
                            "PatientID": patient_id, "StudyUID": study_id, "View": view, "Slice":slice, "CentralSlice": central_slice})
            groups.append(patient_id)
    return dataset, groups

