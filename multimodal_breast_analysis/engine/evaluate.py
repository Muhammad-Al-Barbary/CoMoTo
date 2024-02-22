from monai.transforms import LoadImage
from monai.data import Dataset, DataLoader as MonaiLoader
import os, cv2, torch, numpy as np, natsort, shutil
from multimodal_breast_analysis.engine.utils import Boxes, NMS_volume

import csv
import os
import pandas as pd 
from tqdm import tqdm 


def evaluate(
    labels_fp,
    boxes_fp,
    predictions_fp,
):
    """Evaluate predictions"""
    df_labels = pd.read_csv(labels_fp)
    df_boxes = pd.read_csv(boxes_fp, dtype={"VolumeSlices": float})
    df_pred = pd.read_csv(predictions_fp, dtype={"Score": float})

    df_labels = df_labels.reset_index().set_index(["StudyUID", "View"]).sort_index()
    df_boxes = df_boxes.reset_index().set_index(["StudyUID", "View"]).sort_index()
    df_pred = df_pred.reset_index().set_index(["StudyUID", "View"]).sort_index()

    df_pred["TP"] = 0
    df_pred["GTID"] = -1

    thresholds = [df_pred["Score"].max() + 1.0]

    # find true positive predictions and assign detected ground truth box ID
    for box_pred in df_pred.itertuples():
        if box_pred.Index not in df_boxes.index:
            continue

        df_boxes_view = df_boxes.loc[[box_pred.Index]]
        view_slice_offset = df_boxes.loc[[box_pred.Index], "VolumeSlices"].iloc[0] / 4
        tp_boxes = [
            b
            for b in df_boxes_view.itertuples()
            if _is_tp(box_pred, b, slice_offset=view_slice_offset)
        ]
        if len(tp_boxes) > 1:
            # find the nearest GT box
            tp_distances = [_distance(box_pred, b) for b in tp_boxes]
            tp_boxes = [tp_boxes[np.argmin(tp_distances)]]
        if len(tp_boxes) > 0:
            tp_i = tp_boxes[0].index
            df_pred.loc[df_pred["index"] == box_pred.index, ("TP", "GTID")] = (1, tp_i)
            thresholds.append(box_pred.Score)

    thresholds.append(df_pred["Score"].min() - 1.0)

    # compute sensitivity at 2 FPs/volume on all cases
    evaluation_fps_all = (2.0,)
    tpr_all = _froc(
        df_pred=df_pred,
        thresholds=thresholds,
        n_volumes=len(df_labels),
        n_boxes=len(df_boxes),
        evaluation_fps=evaluation_fps_all,
    )
    result = {f"sensitivity_at_2_fps_all": tpr_all[0]}

    # compute mean sensitivity at 1, 2, 3, 4 FPs/volume on positive cases
    df_pred = df_pred[df_pred.index.isin(df_boxes.index)]
    df_labels = df_labels[df_labels.index.isin(df_boxes.index)]
    evaluation_fps_positive = (1.0, 2.0, 3.0, 4.0)
    tpr_positive = _froc(
        df_pred=df_pred,
        thresholds=thresholds,
        n_volumes=len(df_labels),
        n_boxes=len(df_boxes),
        evaluation_fps=evaluation_fps_positive,
    )

    result.update(
        dict(
            (f"sensitivity_at_{int(x)}_fps_positive", y)
            for x, y in zip(evaluation_fps_positive, tpr_positive)
        )
    )
    result.update({"mean_sensitivity_positive": np.mean(tpr_positive)})

    return result


def _froc(
    df_pred: pd.DataFrame,
    thresholds,
    n_volumes: int,
    n_boxes: int,
    evaluation_fps: tuple,
):
    tpr = []
    fps = []
    for th in sorted(thresholds, reverse=True):
        df_th = df_pred.loc[df_pred["Score"] >= th]
        df_th_unique_tp = df_th.reset_index().drop_duplicates(
            subset=["StudyUID", "View", "TP", "GTID"]
        )
        n_tps_th = float(sum(df_th_unique_tp["TP"]))
        tpr_th = n_tps_th / n_boxes
        n_fps_th = float(len(df_th[df_th["TP"] == 0]))
        fps_th = n_fps_th / n_volumes
        tpr.append(tpr_th)
        fps.append(fps_th)
        if fps_th > max(evaluation_fps):
            break
    return [np.interp(x, fps, tpr) for x in evaluation_fps]


def _is_tp(
    box_pred, box_true, slice_offset: int, min_dist: int = 100
) -> bool:
    pred_y = box_pred.Y + box_pred.Height / 2
    pred_x = box_pred.X + box_pred.Width / 2
    pred_z = box_pred.Z + box_pred.Depth / 2

    true_y = (box_true.ymin + box_true.ymax)/2
    true_x = (box_true.xmin + box_true.xmax)/2
    true_z = box_true.Slice
    Height =  box_true.ymax - box_true.ymin
    Width =  box_true.xmax - box_true.xmin
    # 2D distance between true and predicted center points
    dist = np.linalg.norm((pred_x - true_x, pred_y - true_y))
    # compute radius based on true box size
    dist_threshold = np.sqrt(Width ** 2 + Height ** 2) / 2.0
    dist_threshold = max(dist_threshold, min_dist)
    slice_diff = np.abs(pred_z - true_z)
    # TP if predicted center within radius and slice within slice offset
    return dist <= dist_threshold and slice_diff <= slice_offset


def _distance(box_pred, box_true) -> float:
    pred_y = box_pred.Y + box_pred.Height / 2
    pred_x = box_pred.X + box_pred.Width / 2
    pred_z = box_pred.Z + box_pred.Depth / 2
    true_y = (box_true.ymin + box_true.ymax)/2
    true_x = (box_true.xmin + box_true.xmax)/2
    true_z = box_true.Slice
    return np.linalg.norm((pred_x - true_x, pred_y - true_y, pred_z - true_z))




def write_csv (final_boxes_vol, final_scores_vol, final_slices_vol, client, episode, view, total_slices, output_path = 'output_folder/', output_csv = 'test_results.csv'):
    depth = 0
    with open(output_path+output_csv, 'a+', newline='') as file:
        writer = csv.writer(file)
        for box, score,slice_num in zip(final_boxes_vol,final_scores_vol, final_slices_vol):

            writer.writerow([client, episode, view, int(box[0]),int(box[2]-box[0]), int(box[1]), int(box[3]-box[1]),
                  max(0,slice_num-depth),  min(total_slices-1, slice_num+depth)-max(0,slice_num-depth),float(score)])



def dbt_final_eval(engine, metadata_path = None, output_path = 'output_folder/', output_csv = 'test_results.csv', target_csv = 'targets.csv', temp_path="pred_temp_folder/"):
    if metadata_path is None:
        metadata_path = engine.config.data['student_args'][1]
    df = pd.read_csv(metadata_path)
    if os.path.exists(output_path+target_csv):
        os.remove(output_path+target_csv)
    if os.path.exists(output_path+output_csv):
        os.remove(output_path+output_csv)

    # target csv
    with open(output_path+output_csv, 'w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['PatientID','StudyUID','View','X','Width','Y','Height','Z','Depth','Score'])
    test_clients = [engine.student_testloader.dataset[i]['PatientID'] for i in range(len(engine.student_testloader.dataset))]
    target_df = df[df['PatientID'].isin(test_clients)]
    target_df.to_csv(output_path+target_csv, index=False) 
    
    # prediction csv
    target_df = target_df.drop_duplicates(subset='path') # predict each volume only once
    for index, view_series in tqdm(target_df.iterrows()):
        client = view_series["PatientID"]
        view = view_series["View"]
        image_path = view_series["path"]
        episode = view_series["StudyUID"]
        num_slices = view_series["VolumeSlices"]
        final_boxes_vol, final_scores_vol, final_slices_vol = engine.predict_2dto3d(image_path, temp_path = temp_path)
        write_csv(final_boxes_vol, final_scores_vol, final_slices_vol, client, episode, view, num_slices, output_path = 'output_folder/', output_csv = 'test_results.csv')
    results = evaluate(labels_fp = output_path+target_csv, boxes_fp = output_path+target_csv, predictions_fp = output_path+output_csv)
    return results