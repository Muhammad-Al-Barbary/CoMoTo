from multimodal_breast_analysis.engine.engine import Engine
from multimodal_breast_analysis.configs.configs import load_configs
from multimodal_breast_analysis.engine.utils import prepare_batch, closest_index 
from multimodal_breast_analysis.engine.evaluate import dbt_final_eval
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
        predictions = np.concatenate([results_metric[i][1]['dtScores'] for i in range(len(results_metric))],0)
        targets = np.concatenate([results_metric[i][1]['dtMatches'][0] for i in range(len(results_metric))], 0)
        num_imgs = len([results_metric[i][1]['dtMatches'][0] for i in range(len(results_metric))])
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(targets, predictions)
        roc_auc = auc(fpr, tpr)
        print("AUC:", roc_auc)
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
        print("Mean Sensitivity:", mean_sensitivity)
        print("Sensitivity@2FPI:", sensitivity_at2fpi)



def main(args):
    config = load_configs(args.config_name)
    engine = Engine(config)
    
    if args.mammo:
        engine.warmup()
        engine.load(mode="teacher", path=config.networks["best_teacher_cp"])
        print()
        print("Final Validation:", engine.test('teacher', 'validation'))
        print()
        print("Final Testing:", engine.test('teacher', 'testing'))
        print()
        mammo_final_eval(engine)

    if args.dbt:
        engine.load(mode="teacher", path=config.networks["best_teacher_cp"])
        engine.load(mode="student", path=config.networks["best_teacher_cp"])
        engine.train()
        engine.load(mode="student", path=config.networks["best_student_cp"])
        print("\n")
        print("Final Validation:", engine.test('student', 'validation'))
        print("\n")
        print("Final Testing:", engine.test('student', 'testing'))
        print('\n\n')
        print("Final Metrics", dbt_final_eval(engine))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Multimodal Breast Lesion Detection')
    parser.add_argument(
                '--config_name', type = str, default = None,
                help = 'name of the config file to be loaded (without extension)'
                )
    parser.add_argument(
                '--mammo', action='store_true',
                help = 'train mammography model'
                )
    parser.add_argument(
                '--dbt', action='store_true',
                help = 'train dbt model'
                )
    args = parser.parse_args()
    main(args)