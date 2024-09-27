"""
DBT evaluation script refactored from: https://github.com/mazurowski-lab/duke-dbt-data/blob/master/duke_dbt_data.py
"""
from multimodal_breast_analysis.engine.utils import prepare_batch 

import os
import torch
import numpy as np
import csv
import os
import pandas as pd 
from tqdm import tqdm 
import gc
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from monai.data.box_utils import box_iou
from monai.apps.detection.metrics.matching import matching_batch



def mammo_final_eval(engine, loader_mode = 'testing'):
    threshold = [0.1]
    metrics = {}
    if loader_mode=='training':
        dataloader = engine.teacher_trainloader
    elif loader_mode=='validation':
        dataloader = engine.teacher_validloader
    elif loader_mode=='testing':
        dataloader = engine.teacher_testloader            
    network = engine.teacher
    print("Mammography final evaluation on", loader_mode, 'set')
    network.eval()
    with torch.no_grad():
        all_targets = []
        all_predictions = []
        for batch_num, batch in enumerate(tqdm(dataloader, unit="iter")):
            gc.collect()
            torch.cuda.empty_cache()
            images, targets = prepare_batch(batch, engine.device)
            predictions = network(images)
            all_targets += targets
            all_predictions += predictions
        results_metric = matching_batch(
            iou_fn=box_iou,
            iou_thresholds=threshold,
            pred_boxes=[sample["boxes"].cpu().numpy() for sample in all_predictions],
            pred_classes=[sample["labels"].cpu().numpy() for sample in all_predictions],
            pred_scores=[sample["scores"].cpu().numpy() for sample in all_predictions],
            gt_boxes=[sample["boxes"].cpu().numpy() for sample in all_targets],
            gt_classes=[sample["labels"].cpu().numpy() for sample in all_targets],
        )
        predictions = np.concatenate([results_metric[i][1]['dtScores'] for i in range(len(results_metric))],0)
        targets = np.concatenate([results_metric[i][1]['dtMatches'][0] for i in range(len(results_metric))], 0)
        num_imgs = len([results_metric[i][1]['dtMatches'][0] for i in range(len(results_metric))])
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(targets, predictions)
        roc_auc = auc(fpr, tpr)
        metrics['auc'] = roc_auc
        all_thresholds = predictions.copy()
        all_thresholds.sort()
        fpis = []
        sensitivities = []
        sensitivities_at1fpi = []
        sensitivities_at2fpi = []
        sensitivities_at3fpi = []
        sensitivities_at4fpi = []
        for threshold in all_thresholds:
            conf_matrix = confusion_matrix(targets, predictions>=threshold, labels=[0,1])
            tn, fp, fn, tp = conf_matrix.ravel()
            fpi = fp/num_imgs
            fpis.append(fpi)
            sensitivity = tp / (tp + fn)
            sensitivities.append(sensitivity)
            if fpi == 1:
                sensitivities_at1fpi.append(sensitivity)
            elif fpi == 2:
                sensitivities_at2fpi.append(sensitivity)
            elif fpi == 3:
                sensitivities_at3fpi.append(sensitivity)
            elif fpi == 4:
                sensitivities_at4fpi.append(sensitivity)
        sensitivity_at1fpi = sum(sensitivities_at1fpi)/len(sensitivities_at1fpi)
        sensitivity_at2fpi = sum(sensitivities_at2fpi)/len(sensitivities_at2fpi)
        sensitivity_at3fpi = sum(sensitivities_at3fpi)/len(sensitivities_at3fpi)
        sensitivity_at4fpi = sum(sensitivities_at4fpi)/len(sensitivities_at4fpi)
        mean_sensitivity = (sensitivity_at1fpi + sensitivity_at2fpi + sensitivity_at3fpi + sensitivity_at4fpi) / 4
        metrics['mean_sensitivity'] = mean_sensitivity
        metrics['sensitivity_@2fps'] = sensitivity_at2fpi
        return  metrics

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


def write_csv (final_boxes_vol, final_scores_vol, final_slices_vol, client, episode, view, total_slices, output_path = 'output_folder/', pred_csv = 'test_results.csv'):
    depth = 0
    rows_to_write = []
    for box, score, slice_num in zip(final_boxes_vol, final_scores_vol, final_slices_vol):
        row = [client, episode, view, int(box[0]), int(box[2] - box[0]), int(box[1]), int(box[3] - box[1]),
            max(0, slice_num - depth), min(total_slices - 1, slice_num + depth) - max(0, slice_num - depth), float(score)]
        rows_to_write.append(row)
    with open(output_path + pred_csv, 'a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows_to_write)


def dbt_final_eval(engine, metadata_path = None, output_path = 'output_folder/', pred_csv = 'test_results.csv', target_csv = 'targets.csv', temp_path="pred_temp_folder/"):
    if metadata_path is None: #TODO: Fix this hardcoding
        metadata_path = "../datasets/dbt/metadata_test.csv"
    # target csv
        df = pd.read_csv(metadata_path)
    if not os.path.exists(output_path+target_csv):
        target_df = df
        target_df.to_csv(output_path+target_csv, index=False)
    else:
        target_df = pd.read_csv(output_path+target_csv)
    # prediction csv
    if os.path.exists(output_path+pred_csv):
        os.remove(output_path+pred_csv)
    with open(output_path+pred_csv, 'w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['PatientID','StudyUID','View','X','Width','Y','Height','Z','Depth','Score'])
    target_df = target_df.drop_duplicates(subset='path') # predict each volume only once
    for index, view_series in tqdm(target_df.iterrows(), total=len(target_df)):
        client = view_series["PatientID"]
        view = view_series["View"]
        image_path = view_series["path"]
        episode = view_series["StudyUID"]
        num_slices = int(view_series["VolumeSlices"])
        final_boxes_vol, final_scores_vol, final_slices_vol = engine.predict_2dto3d(image_path, temp_path = temp_path)
        write_csv(final_boxes_vol, final_scores_vol, final_slices_vol, client, episode, view, num_slices, output_path = 'output_folder/', pred_csv = pred_csv)
    results = evaluate(labels_fp = output_path+target_csv, boxes_fp = output_path+target_csv, predictions_fp = output_path+pred_csv)
    return results