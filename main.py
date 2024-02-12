from multimodal_breast_analysis.engine.engine import Engine
from multimodal_breast_analysis.configs.configs import load_configs
from multimodal_breast_analysis.engine.utils import prepare_batch, closest_index 

import os
import argparse
import torch
import gc
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from monai.data.box_utils import box_iou
from monai.apps.detection.metrics.matching import matching_batch



def mammo_final_eval(engine, loader_mode = 'testing'):
    threshold = [0.1]
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
        targets_perimg = [results_metric[i][1]['dtMatches'][0] for i in range(len(results_metric))]
        predictions_perimg = [results_metric[i][1]['dtScores'] for i in range(len(results_metric))]
        predictions, targets = np.concatenate(predictions_perimg,0), np.concatenate(targets_perimg, 0)
        fpr, tpr, thresholds = roc_curve(targets, predictions)
        roc_auc = auc(fpr, tpr)
        print("AUC:", roc_auc)
        thresholds = []
        FPIs = []
        temp = predictions.copy()
        temp.sort()
        for threshold in temp:
            fps_all = []
            for i in range(len(predictions_perimg)):
                conf_matrix = confusion_matrix(targets_perimg[i], predictions_perimg[i]>=threshold, labels=[0,1])
                _, false_positives, _, _ = conf_matrix.ravel()
                fps_all.append(false_positives)
            thresholds.append(threshold)
            FPIs.append(sum(fps_all)/len(fps_all)) 
        threshold1 = thresholds[closest_index(FPIs, 1)]
        threshold2 = thresholds[closest_index(FPIs, 2)]
        threshold3 = thresholds[closest_index(FPIs, 3)]
        threshold4 = thresholds[closest_index(FPIs, 4)]
        mean_sensitivity = 0
        for threshold in [threshold1,threshold2,threshold3,threshold4]:
            conf_matrix = confusion_matrix(np.asarray(targets), np.asarray(predictions)>threshold, labels=[0,1])
            true_negatives, false_positives, false_negatives, true_positives = conf_matrix.ravel()
            sensitivity = true_positives / (true_positives + false_negatives)
            mean_sensitivity += sensitivity
        mean_sensitivity/=4
        print("Mean Sensitivity:", mean_sensitivity)
        conf_matrix = confusion_matrix(np.asarray(targets), np.asarray(predictions)>threshold2, labels=[0,1])
        true_negatives, false_positives, false_negatives, true_positives = conf_matrix.ravel()
        sensitivity = true_positives / (true_positives + false_negatives)
        print("Sensitivity @ 2FPI:", sensitivity)



def dbt_final_eval(engine):
    dataset_path = '/home/muhammad/multimodal_learning/datasets/dbt/3d/'
    metadata_path = '/home/muhammad/multimodal_learning/datasets/dbt/metadata.csv'
    metadata = pd.read_csv(metadata_path)
    predictions_all = []
    targets_all = []
    for idx, path in enumerate(os.listdir(dataset_path)):
        volume_path = dataset_path + path
        patient = path.split('_')[0]
        if patient not in [engine.student_testloader.dataset[i]['PatientID'] for i in range(len(engine.student_testloader.dataset))]:
            continue
        study = path.split('_')[1]
        view = path.split('_')[2].split('.')[0]
        rows = metadata[(metadata['PatientID'] == patient) & (metadata['StudyUID'] == study) & (metadata['View'] == view)]
        xmin, ymin, xmax, ymax = list(rows['xmin']), list(rows['ymin']), list(rows['xmax']), list(rows['ymax'])
        slices = list(rows['Slice'])
        volume_slices = list(rows['VolumeSlices'])[0]
        zmin = [max(0, slice - int(0.25 * volume_slices)) for slice in slices]
        zmax = [min(slice + int(0.25 * volume_slices), volume_slices-1) for slice in slices]
        true_boxes = torch.stack([torch.tensor([xmin[i], ymin[i], zmin[i], xmax[i], ymax[i], zmax[i]]) for i in range(len(xmin))])
        targets_all += [{'boxes' : true_boxes, 'labels' : torch.ones(true_boxes.shape[0])}]
        #PREDICTION
        final_boxes_vol ,final_scores_vol = engine.predict_2dto3d(volume_path)
        predictions_all += [{'boxes' : final_boxes_vol, 'scores' : final_scores_vol, 'labels' : torch.ones(final_scores_vol.shape)}]
    #Evaluate
    threshold = [0.1]
    results_metric = matching_batch(
        iou_fn=box_iou,
        iou_thresholds=threshold,
        pred_boxes=[sample["boxes"].cpu().float().numpy() for sample in predictions_all],
        pred_classes=[sample["labels"].cpu().float().numpy() for sample in predictions_all],
        pred_scores=[sample["scores"].cpu().float().numpy() for sample in predictions_all],
        gt_boxes=[sample["boxes"].cpu().float().numpy() for sample in targets_all],
        gt_classes=[sample["labels"].cpu().float().numpy() for sample in targets_all],
    )
    targets = np.concatenate([results_metric[i][1]['dtMatches'][0] for i in range(len(results_metric))], axis = 0)
    predictions = np.concatenate([results_metric[i][1]['dtScores'] for i in range(len(results_metric))], axis = 0)
    targets_pervol = [results_metric[i][1]['dtMatches'][0] for i in range(len(results_metric))]
    preds_pervol = [results_metric[i][1]['dtScores'] for i in range(len(results_metric))]
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(targets, predictions)
    roc_auc = auc(fpr, tpr)
    print("AUC:", roc_auc)
    thresholds = []
    FPIs = []
    temp = predictions.copy()
    temp.sort()
    for threshold in temp:
        fps_all = []
        for i in range(len(preds_pervol)):
            conf_matrix = confusion_matrix(targets_pervol[i], preds_pervol[i]>=threshold, labels=[0,1])
            _, false_positives, _, _ = conf_matrix.ravel()
            fps_all.append(false_positives)
        thresholds.append(threshold)
        FPIs.append(sum(fps_all)/len(fps_all)) 
    threshold1 = thresholds[closest_index(FPIs, 1)]
    threshold2 = thresholds[closest_index(FPIs, 2)]
    threshold3 = thresholds[closest_index(FPIs, 3)]
    threshold4 = thresholds[closest_index(FPIs, 4)]
    mean_sensitivity = 0
    for threshold in [threshold1,threshold2,threshold3,threshold4]:
        conf_matrix = confusion_matrix(np.asarray(targets), np.asarray(predictions)>threshold, labels=[0,1])
        true_negatives, false_positives, false_negatives, true_positives = conf_matrix.ravel()
        sensitivity = true_positives / (true_positives + false_negatives)
        mean_sensitivity += sensitivity
    mean_sensitivity/=4
    print("Mean Sensitivity:", mean_sensitivity)
    conf_matrix = confusion_matrix(np.asarray(targets), np.asarray(predictions)>threshold2, labels=[0,1])
    true_negatives, false_positives, false_negatives, true_positives = conf_matrix.ravel()
    sensitivity = true_positives / (true_positives + false_negatives)
    print("Sensitivity @ 2FPI:", sensitivity)




def main(args):
    config = load_configs(args.config_name)
    engine = Engine(config)
    if args.mammo:
        engine.warmup()
        engine.load(mode="teacher", path=config.networks["best_teacher_cp"])
        mammo_final_eval(engine)
    if args.dbt:
        engine.load(mode="teacher", path=config.networks["best_teacher_cp"])
        engine.load(mode="student", path=config.networks["best_teacher_cp"])
        engine.train()
        engine.load(mode="student", path=config.networks["best_student_cp"])
        dbt_final_eval(engine)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Multimodal Breast Lesion Detection')
    parser.add_argument(
                '--config_name', type = str, default = None,
                help = 'name of the config file to be loaded (without extension)'
                )
    parser.add_argument(
                '--mammo', type = bool, default = True,
                help = 'train mammography model'
                )
    parser.add_argument(
                '--dbt', type = bool, default = True,
                help = 'train dbt model'
                )
    args = parser.parse_args()
    main(args)