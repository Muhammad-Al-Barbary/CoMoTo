from multimodal_breast_analysis.models.faster_rcnn import faster_rcnn, faster_rcnn_fpn
from multimodal_breast_analysis.models.unet import UNet as unet
from multimodal_breast_analysis.data.dataloader import DataLoader
from multimodal_breast_analysis.data.transforms import train_transforms, test_transforms
from multimodal_breast_analysis.data.datasets import penn_fudan, omidb, omidb_downsampled, dbt_2d
from multimodal_breast_analysis.engine.utils import prepare_batch, log_transforms, average_dicts, set_seed
from multimodal_breast_analysis.engine.visualization import visualize_batch
from multimodal_breast_analysis.engine.utils import Boxes, NMS_volume

import os
import cv2
import natsort
import shutil
import wandb
import logging
from tqdm import tqdm
import torch
import torchvision
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CyclicLR
from torch.nn import KLDivLoss, Flatten, Linear
from torch.nn.functional import softmax, log_softmax, cosine_similarity
from monai.data.box_utils import box_iou
from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch
from monai.data import Dataset
from monai.data import DataLoader as MonaiLoader
from monai.transforms import LoadImage
import gc
from random import randint
import numpy as np


class Engine:
    def __init__(self, config):
        self.config = config
        set_seed(self.config.seed)
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.student = self._get_model(
            self.config.networks["student"],
            self.config.networks["student_parameters"]
            ).to(self.device)
        self.teacher = self._get_model(
            self.config.networks["teacher"], 
            self.config.networks["teacher_parameters"]
            ).to(self.device)    
        self.student_optimizer = self._get_optimizer(
            self.config.train["student_optimizer"]
        )(self.student.parameters(),**self.config.train["student_optimizer_parameters"])
        self.teacher_optimizer = self._get_optimizer(
            self.config.train["teacher_optimizer"]
        )(self.teacher.parameters(),**self.config.train["teacher_optimizer_parameters"])
        self.student_scheduler = self._get_scheduler(
            self.config.train["student_scheduler"],
        )(self.student_optimizer,**self.config.train["student_scheduler_parameters"])
        self.teacher_scheduler = self._get_scheduler(
            self.config.train["teacher_scheduler"],
        )(self.teacher_optimizer,**self.config.train["teacher_scheduler_parameters"])


        student_loader = DataLoader(
                            data=self._get_data(self.config.data["student_name"])(*self.config.data["student_args"]),
                            valid_split=self.config.data['valid_split'],
                            test_split=self.config.data['test_split'],
                            seed=self.config.seed
                            )
        teacher_loader = DataLoader(
                            data=self._get_data(self.config.data["teacher_name"])(*self.config.data["teacher_args"]),
                            valid_split=self.config.data['valid_split'],
                            test_split=self.config.data['test_split'],
                            seed=self.config.seed
                            )
        self.student_trainloader = student_loader.trainloader(
                                        train_transforms(self.config.data["student_name"], self.config.transforms), 
                                        batch_size=self.config.data["batch_size"], 
                                        shuffle=self.config.data["shuffle"],
                                        train_ratio=self.config.data["train_ratio"],
                                        )
        self.student_validloader = student_loader.validloader(
                                        test_transforms(self.config.data["student_name"], self.config.transforms), 
                                        batch_size=self.config.data["batch_size"], 
                                        )
        self.student_testloader = student_loader.testloader(
                                        test_transforms(self.config.data["student_name"], self.config.transforms), 
                                        batch_size=self.config.data["batch_size"], 
                                        )

        self.teacher_trainloader = teacher_loader.trainloader(
                                        train_transforms(self.config.data["teacher_name"], self.config.transforms), 
                                        batch_size=self.config.data["batch_size"], 
                                        shuffle=self.config.data["shuffle"]
                                        )
        self.teacher_validloader = teacher_loader.validloader(
                                        test_transforms(self.config.data["teacher_name"], self.config.transforms), 
                                        batch_size=self.config.data["batch_size"], 
                                        )
        self.teacher_testloader = teacher_loader.testloader(
                                        test_transforms(self.config.data["teacher_name"], self.config.transforms), 
                                        batch_size=self.config.data["batch_size"], 
                                        )

        student_train_logs, student_test_logs = log_transforms(
                                                                "multimodal_breast_analysis/data/transforms.py", 
                                                                self.config.data["student_name"]
                                                                )
        teacher_train_logs, teacher_test_logs = log_transforms(
                                                                "multimodal_breast_analysis/data/transforms.py", 
                                                                self.config.data["teacher_name"]
                                                                )
        transform_logs = wandb.Table(
                          columns=["Teacher", "Student"], 
                          data=[[teacher_train_logs,student_train_logs], [teacher_test_logs,student_test_logs]]
                          )

        wandb.log({"Transforms": transform_logs})


    def _get_data(self, dataset_name):
        data = {
             "penn_fudan": penn_fudan,
             "omidb": omidb,
             "omidb_downsampled": omidb_downsampled,
             "dbt_2d": dbt_2d
             } #
        return data[dataset_name]

    def _get_model(self, name, parameters):
        models = {
            "faster_rcnn": faster_rcnn,
            "unet": unet,
            "faster_rcnn_fpn" : faster_rcnn_fpn
        }
        return models[name](parameters)
    

    def _get_optimizer(self, name):
        optimizers = {
            "sgd": SGD,
            "adam": Adam
        }
        return optimizers[name]
   
        
    def _get_scheduler(self, name):
        schedulers = {
            "step" : StepLR,
            "cyclic" : CyclicLR,
        }
        return schedulers[name]
    

    def save(self, mode, path=None):
        if mode == 'teacher':
            if path is None:
                path = self.config.networks["last_teacher_cp"]
            checkpoint = {
                "network": self.teacher.state_dict(),
                # "optimizer": self.teacher_optimizer.state_dict(),
                }
        elif mode == 'student':
            if path is None:
                path = self.config.networks["last_student_cp"]
            checkpoint = {
                "network": self.student.state_dict(),
                # "optimizer": self.student_optimizer.state_dict(),
                }            
        torch.save(checkpoint, path)


    def load(self, mode, path=None):
        if mode == "teacher":
            if path is None:
                path = self.config.networks["last_teacher_cp"]
            checkpoint = torch.load(path, map_location=torch.device(self.device))
            self.teacher.load_state_dict(checkpoint['network'])
            # self.teacher_optimizer.load_state_dict(checkpoint['optimizer'])
        elif mode == "student":
            if path is None:
                path = self.config.networks["last_student_cp"]
            checkpoint = torch.load(path, map_location=torch.device(self.device))
            self.student.load_state_dict(checkpoint['network'])
            # self.student_optimizer.load_state_dict(checkpoint['optimizer'])


    def warmup(self):
        CROSS_ALIGN = self.config.train['cross_align'] # not fully functional yet
        if CROSS_ALIGN:
            def cross_align_loss(positive_features, negative_features):
                positive_features = positive_features.view(positive_features.size(0), -1)  
                negative_features = negative_features.view(negative_features.size(0), -1)
                pos_similarity = cosine_similarity(positive_features.unsqueeze(1), positive_features.unsqueeze(0), dim=2)
                neg_similarity = cosine_similarity(positive_features.unsqueeze(1), negative_features.unsqueeze(0), dim=2)
                loss = torch.mean(torch.relu(1 - pos_similarity)) + torch.mean(torch.relu(neg_similarity))
                return self.config.train['cross_align_coeff'] * loss

            self.features = {}
            def get_features(name):
                def hook(model, input, output):
                    self.features[name] = output
                return hook
            self.teacher.backbone.register_forward_hook(get_features('teacher'))

        warmup_epochs = self.config.train["warmup_epochs"]
        best_metric = 0
        for epoch in range(warmup_epochs):
            print(f"\nWarmup Epoch {epoch+1}/{warmup_epochs}\n-------------------------------")
            epoch_total_loss = 0
            epoch_detection_loss = 0
            epoch_similarity_loss = 0
            for batch_num, batch in enumerate(tqdm(self.teacher_trainloader, unit="iter")):
                gc.collect()
                torch.cuda.empty_cache()
                self.teacher.train()
                image, target = prepare_batch(batch, self.device)
                loss = self.teacher(
                            image,
                            target
                            )
                loss = sum(sample_loss for sample_loss in loss.values())
                epoch_detection_loss += loss.item()
##############################################Cross Alignment########################################################################################
                if CROSS_ALIGN:
                    teacher_features = self.features['teacher']
                    teacher_ratio = (teacher_features.shape[-1]-1) / (image[0].shape[-1]-1)
                    teacher_boxes = [((target[i]['boxes']) * teacher_ratio).round().int() for i in range(len(target))]      
                    teacher_features_selected_negative = []
                    for i in range(len(teacher_boxes)):
                        boxes_mask = torch.ones((teacher_features.shape[-2], teacher_features.shape[-1]))
                        for xmin,ymin,xmax,ymax in teacher_boxes[i]:
                            boxes_mask[ymin:ymax, xmin:xmax] = 0  
                        positive_indices = torch.nonzero(boxes_mask == 1)
                        shuffled_indices = positive_indices[torch.randperm(positive_indices.size(0))]
                        sampled_indices = shuffled_indices[:min(9, shuffled_indices.size(0))].t()
                        teacher_features_selected_negative.append(teacher_features[i, :, sampled_indices[0], sampled_indices[1]])
                    teacher_features_selected_negative = torch.stack(teacher_features_selected_negative, dim = 0)

                    teacher_feature_botleft, teacher_feature_botright, teacher_feature_topleft, teacher_feature_topright = [], [], [], []
                    teacher_feature_center, teacher_feature_left, teacher_feature_right, teacher_feature_top, teacher_feature_bot = [], [], [], [], []
                    for sample_num in range(len(teacher_features)):
                        teacher_num_boxes = teacher_boxes[sample_num].shape[0]
                        if teacher_num_boxes > 0:
                            teacher_feature_botleft.extend(
                                [teacher_features[sample_num, :, teacher_boxes[sample_num][box_num][1], teacher_boxes[sample_num][box_num][0]] for box_num in range(teacher_num_boxes)]
                           )
                            teacher_feature_botright.extend(
                                [teacher_features[sample_num, :, teacher_boxes[sample_num][box_num][1], teacher_boxes[sample_num][box_num][2]] for box_num in range(teacher_num_boxes)]
                           )
                            teacher_feature_topleft.extend(
                                [teacher_features[sample_num, :, teacher_boxes[sample_num][box_num][3], teacher_boxes[sample_num][box_num][0]] for box_num in range(teacher_num_boxes)]
                           )
                            teacher_feature_topright.extend(
                                [teacher_features[sample_num, :, teacher_boxes[sample_num][box_num][3], teacher_boxes[sample_num][box_num][2]] for box_num in range(teacher_num_boxes)]
                           )
                            teacher_feature_center.extend(
                                [teacher_features[sample_num, :, int((teacher_boxes[sample_num][box_num][1]+teacher_boxes[sample_num][box_num][3])/2), int((teacher_boxes[sample_num][box_num][0]+teacher_boxes[sample_num][box_num][2])/2)] for box_num in range(teacher_num_boxes)]
                           )
                            teacher_feature_left.extend(
                                [teacher_features[sample_num, :, int((teacher_boxes[sample_num][box_num][1]+teacher_boxes[sample_num][box_num][3])/2), teacher_boxes[sample_num][box_num][0]] for box_num in range(teacher_num_boxes)]
                           )
                            teacher_feature_right.extend(
                                [teacher_features[sample_num, :, int((teacher_boxes[sample_num][box_num][1]+teacher_boxes[sample_num][box_num][3])/2), teacher_boxes[sample_num][box_num][2]] for box_num in range(teacher_num_boxes)]
                           )
                            teacher_feature_top.extend(
                                [teacher_features[sample_num, :, teacher_boxes[sample_num][box_num][1], int((teacher_boxes[sample_num][box_num][0]+teacher_boxes[sample_num][box_num][2])/2)] for box_num in range(teacher_num_boxes)]
                           )
                            teacher_feature_bot.extend(
                                [teacher_features[sample_num, :, teacher_boxes[sample_num][box_num][3], int((teacher_boxes[sample_num][box_num][0]+teacher_boxes[sample_num][box_num][2])/2)] for box_num in range(teacher_num_boxes)]
                           )
                    teacher_feature_botleft = torch.stack(teacher_feature_botleft)
                    teacher_feature_botright = torch.stack(teacher_feature_botright)
                    teacher_feature_topleft = torch.stack(teacher_feature_topleft)
                    teacher_feature_topright = torch.stack(teacher_feature_topright)
                    teacher_feature_center = torch.stack(teacher_feature_center)
                    teacher_feature_left = torch.stack(teacher_feature_left)
                    teacher_feature_right = torch.stack(teacher_feature_right)
                    teacher_feature_top = torch.stack(teacher_feature_top)
                    teacher_feature_bot = torch.stack(teacher_feature_bot)
                    teacher_features_selected_positive = torch.stack(
                        [teacher_feature_botleft, teacher_feature_botright, teacher_feature_topleft, teacher_feature_topright, teacher_feature_center,
                          teacher_feature_left, teacher_feature_right, teacher_feature_top, teacher_feature_bot],
                          dim = -1
                          )
                    sim_loss = cross_align_loss(teacher_features_selected_positive, teacher_features_selected_negative)
                    epoch_similarity_loss += sim_loss.item()
                    loss = loss + sim_loss
                    epoch_total_loss += loss.item()
######################################################################################################################################

                self.teacher_optimizer.zero_grad()
                loss.backward()
                self.teacher_optimizer.step()
                # if not (batch_num % 30):
                #     visualize_batch(self.teacher, image, target, self.config.networks["teacher_parameters"]["classes_names"][1], figsize = (7,7))
            self.teacher_scheduler.step()
            epoch_detection_loss = epoch_detection_loss / len(self.teacher_trainloader)
            epoch_similarity_loss = epoch_similarity_loss / len(self.teacher_trainloader)
            epoch_total_loss = epoch_total_loss / len(self.teacher_trainloader)
            current_metrics = self.test('teacher')
            print("teacher_total_loss:", epoch_total_loss, "detection:", epoch_detection_loss, "similarity:", epoch_similarity_loss)   
            print(current_metrics)
            # for logging in a single dictionary
            current_metrics["teacher_loss"] = epoch_total_loss
            wandb.log(current_metrics)
            self.save('teacher', path = self.config.networks["last_teacher_cp"])
            for k in current_metrics:
                if "mAP" in k:
                    current_metric = current_metrics[k]
            if current_metric >= best_metric:
                best_metric = current_metric
                print('saving best teacher checkpoint')
                self.save('teacher', path = self.config.networks["best_teacher_cp"])


    def distill_loss(self, student_outputs, teacher_outputs, alpha = None, T = None):
        if alpha is None:
            alpha = self.config.train["alpha"]
        if T is None:
            T = self.config.train["temperature"]
        return KLDivLoss(reduction='batchmean')(
            log_softmax(student_outputs/T, dim=1),
            softmax(teacher_outputs/T, dim=1)
            ) * (alpha * T * T)
    

    def _instantiate_kd(self):
        self.features = {}
        def get_features(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook
        self.student.backbone.register_forward_hook(get_features('student'))
        self.teacher.backbone.register_forward_hook(get_features('teacher'))
        with torch.no_grad():
            if self.config.train['distill_mode'] == "image_level":
                student_batch = next(iter(self.student_trainloader))
                teacher_batch = next(iter(self.teacher_trainloader))
                student_image, student_target = prepare_batch(student_batch, self.device)
                teacher_image, teacher_target = prepare_batch(teacher_batch, self.device)
                self.student(student_image, student_target)
                self.teacher(teacher_image, teacher_target)
                student_features = self.features['student']
                teacher_features = self.features['teacher']
                self.flat = Flatten(start_dim=2).to(self.device)
                self.project = Linear(
                                    torch.prod(torch.tensor(student_features.shape[2:])),
                                    torch.prod(torch.tensor(teacher_features.shape[2:]))
                                    ).to(self.device)
                self.student_optimizer.add_param_group({'params':self.project.parameters()})
            elif self.config.train['distill_mode'] == "object_level":
                self.project_selected = Linear(9, 9).to(self.device)
                self.student_optimizer.add_param_group({'params':self.project_selected.parameters()})

    def train(self):
        self._instantiate_kd()
        epochs = self.config.train["epochs"]
        distill_epoch = self.config.train["distill_epoch"]
        distill_loss = torch.tensor(0)
        best_metric = 0 
        teacher_iter = iter(self.teacher_trainloader)
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}\n-------------------------------")
            epoch_total_loss = 0
            epoch_base_loss = 0
            epoch_distill_loss = 0
            self.student.train()
            self.teacher.eval()
            for batch_num, student_batch in enumerate(tqdm(self.student_trainloader, unit="iter")):
                gc.collect()
                torch.cuda.empty_cache()
                student_image, student_target = prepare_batch(student_batch, self.device)
                base_loss = self.student(
                                    student_image,
                                    student_target
                                    )
                base_loss = sum(sample_loss for sample_loss in base_loss.values())
                if epoch >= distill_epoch:
                    try:
                        teacher_batch = next(teacher_iter)
                    except StopIteration:
                        teacher_iter = iter(self.teacher_trainloader)
                        teacher_batch = next(teacher_iter)
                    teacher_image, teacher_target = prepare_batch(teacher_batch, self.device)
                    with torch.no_grad():
                        self.teacher(teacher_image)
                    teacher_features = self.features['teacher']
                    student_features = self.features['student']

                    ##################### Lesion-specific KD ####################################################
                    if self.config.train['distill_mode'] == "object_level":
                        student_ratio = (student_features.shape[-1]-1) / (student_image[0].shape[-1]-1)
                        teacher_ratio = (teacher_features.shape[-1]-1) / (teacher_image[0].shape[-1]-1)
                        student_boxes = [(student_target[i]['boxes'] * student_ratio).round().int() for i in range(len(student_target))]
                        teacher_boxes = [(teacher_target[i]['boxes'] * teacher_ratio).round().int() for i in range(len(teacher_target))]

                        student_features_selected_negative = []
                        for i in range(len(student_boxes)):
                            boxes_mask = torch.ones((student_features.shape[-2], student_features.shape[-1]))
                            for xmin,ymin,xmax,ymax in student_boxes[i]:
                                boxes_mask[ymin:ymax, xmin:xmax] = 0  
                            positive_indices = torch.nonzero(boxes_mask == 1)
                            shuffled_indices = positive_indices[torch.randperm(positive_indices.size(0))]
                            sampled_indices = shuffled_indices[:min(9, shuffled_indices.size(0))].t()
                            student_features_selected_negative.append(student_features[i, :, sampled_indices[0], sampled_indices[1]])
                        student_features_selected_negative = torch.stack(student_features_selected_negative, dim = 0).mean(0)
                        student_features_selected_negative = self.project_selected(student_features_selected_negative)
                        
                        student_feature_botleft, student_feature_botright, student_feature_topleft, student_feature_topright = [], [], [], []
                        student_feature_center = []
                        student_feature_left, student_feature_right, student_feature_top, student_feature_bot = [], [], [], []
                        for sample_num in range(len(student_features)):
                            student_num_boxes = student_boxes[sample_num].shape[0]
                            if student_num_boxes > 0:
                                student_feature_botleft.append(sum(
                                    [student_features[sample_num, :, student_boxes[sample_num][box_num][1], student_boxes[sample_num][box_num][0]] for box_num in range(student_num_boxes)]
                                ) / student_num_boxes)
                                student_feature_botright.append(sum(
                                    [student_features[sample_num, :, student_boxes[sample_num][box_num][1], student_boxes[sample_num][box_num][2]] for box_num in range(student_num_boxes)]
                                ) / student_num_boxes)
                                student_feature_topleft.append(sum(
                                    [student_features[sample_num, :, student_boxes[sample_num][box_num][3], student_boxes[sample_num][box_num][0]] for box_num in range(student_num_boxes)]
                                ) / student_num_boxes)
                                student_feature_topright.append(sum(
                                    [student_features[sample_num, :, student_boxes[sample_num][box_num][3], student_boxes[sample_num][box_num][2]] for box_num in range(student_num_boxes)]
                                ) / student_num_boxes)
                                student_feature_center.append(sum(
                                    [student_features[sample_num, :, int((student_boxes[sample_num][box_num][1]+student_boxes[sample_num][box_num][3])/2), int((student_boxes[sample_num][box_num][0]+student_boxes[sample_num][box_num][2])/2)] for box_num in range(student_num_boxes)]
                                ) / student_num_boxes)
                                student_feature_left.append(sum(
                                    [student_features[sample_num, :, int((student_boxes[sample_num][box_num][1]+student_boxes[sample_num][box_num][3])/2), student_boxes[sample_num][box_num][0]] for box_num in range(student_num_boxes)]
                                ) / student_num_boxes)
                                student_feature_right.append(sum(
                                    [student_features[sample_num, :, int((student_boxes[sample_num][box_num][1]+student_boxes[sample_num][box_num][3])/2), student_boxes[sample_num][box_num][2]] for box_num in range(student_num_boxes)]
                                ) / student_num_boxes)
                                student_feature_top.append(sum(
                                    [student_features[sample_num, :, student_boxes[sample_num][box_num][1], int((student_boxes[sample_num][box_num][0]+student_boxes[sample_num][box_num][2])/2)] for box_num in range(student_num_boxes)]
                                ) / student_num_boxes)
                                student_feature_bot.append(sum(
                                    [student_features[sample_num, :, student_boxes[sample_num][box_num][3], int((student_boxes[sample_num][box_num][0]+student_boxes[sample_num][box_num][2])/2)] for box_num in range(student_num_boxes)]
                                ) / student_num_boxes)
                        student_feature_botleft = sum(student_feature_botleft) / len(student_feature_botleft)
                        student_feature_botright = sum(student_feature_botright) / len(student_feature_botright)
                        student_feature_topleft = sum(student_feature_topleft) / len(student_feature_topleft)
                        student_feature_topright = sum(student_feature_topright) / len(student_feature_topright)
                        student_feature_center = sum(student_feature_center) / len(student_feature_center)
                        student_feature_left = sum(student_feature_left) / len(student_feature_left)
                        student_feature_right = sum(student_feature_right) / len(student_feature_right)
                        student_feature_top = sum(student_feature_top) / len(student_feature_top)
                        student_feature_bot = sum(student_feature_bot) / len(student_feature_bot)
                        teacher_feature_botleft, teacher_feature_botright, teacher_feature_topleft, teacher_feature_topright = [], [], [], []
                        teacher_feature_center = []
                        teacher_feature_left, teacher_feature_right, teacher_feature_top, teacher_feature_bot = [], [], [], []
                        for sample_num in range(len(teacher_features)):
                            teacher_num_boxes = teacher_boxes[sample_num].shape[0]
                            if teacher_num_boxes > 0:
                                teacher_feature_botleft.append(sum(
                                    [teacher_features[sample_num, :, teacher_boxes[sample_num][box_num][1], teacher_boxes[sample_num][box_num][0]] for box_num in range(teacher_num_boxes)]
                                ) / teacher_num_boxes)
                                teacher_feature_botright.append(sum(
                                    [teacher_features[sample_num, :, teacher_boxes[sample_num][box_num][1], teacher_boxes[sample_num][box_num][2]] for box_num in range(teacher_num_boxes)]
                                ) / teacher_num_boxes)
                                teacher_feature_topleft.append(sum(
                                    [teacher_features[sample_num, :, teacher_boxes[sample_num][box_num][3], teacher_boxes[sample_num][box_num][0]] for box_num in range(teacher_num_boxes)]
                                ) / teacher_num_boxes)
                                teacher_feature_topright.append(sum(
                                    [teacher_features[sample_num, :, teacher_boxes[sample_num][box_num][3], teacher_boxes[sample_num][box_num][2]] for box_num in range(teacher_num_boxes)]
                                ) / teacher_num_boxes)
                                teacher_feature_center.append(sum(
                                    [teacher_features[sample_num, :, int((teacher_boxes[sample_num][box_num][1]+teacher_boxes[sample_num][box_num][3])/2), int((teacher_boxes[sample_num][box_num][0]+teacher_boxes[sample_num][box_num][2])/2)] for box_num in range(teacher_num_boxes)]
                                ) / teacher_num_boxes)
                                teacher_feature_left.append(sum(
                                    [teacher_features[sample_num, :, int((teacher_boxes[sample_num][box_num][1]+teacher_boxes[sample_num][box_num][3])/2), teacher_boxes[sample_num][box_num][0]] for box_num in range(teacher_num_boxes)]
                                ) / teacher_num_boxes)
                                teacher_feature_right.append(sum(
                                    [teacher_features[sample_num, :, int((teacher_boxes[sample_num][box_num][1]+teacher_boxes[sample_num][box_num][3])/2), teacher_boxes[sample_num][box_num][2]] for box_num in range(teacher_num_boxes)]
                                ) / teacher_num_boxes)
                                teacher_feature_top.append(sum(
                                    [teacher_features[sample_num, :, teacher_boxes[sample_num][box_num][1], int((teacher_boxes[sample_num][box_num][0]+teacher_boxes[sample_num][box_num][2])/2)] for box_num in range(teacher_num_boxes)]
                                ) / teacher_num_boxes)
                                teacher_feature_bot.append(sum(
                                    [teacher_features[sample_num, :, teacher_boxes[sample_num][box_num][3], int((teacher_boxes[sample_num][box_num][0]+teacher_boxes[sample_num][box_num][2])/2)] for box_num in range(teacher_num_boxes)]
                                ) / teacher_num_boxes)
                        teacher_feature_botleft = sum(teacher_feature_botleft) / len(teacher_feature_botleft)
                        teacher_feature_botright = sum(teacher_feature_botright) / len(teacher_feature_botright)
                        teacher_feature_topleft = sum(teacher_feature_topleft) / len(teacher_feature_topleft)
                        teacher_feature_topright = sum(teacher_feature_topright) / len(teacher_feature_topright)
                        teacher_feature_center = sum(teacher_feature_center) / len(teacher_feature_center)
                        teacher_feature_left = sum(teacher_feature_left) / len(teacher_feature_left)
                        teacher_feature_right = sum(teacher_feature_right) / len(teacher_feature_right)
                        teacher_feature_top = sum(teacher_feature_top) / len(teacher_feature_top)
                        teacher_feature_bot = sum(teacher_feature_bot) / len(teacher_feature_bot)
                        teacher_features_selected = torch.stack([teacher_feature_botleft, teacher_feature_botright, teacher_feature_topleft, teacher_feature_topright, teacher_feature_center, teacher_feature_left, teacher_feature_right, teacher_feature_top, teacher_feature_bot]).permute(1,0)
                        student_features_selected = torch.stack([student_feature_botleft, student_feature_botright, student_feature_topleft, student_feature_topright, student_feature_center, student_feature_left, student_feature_right, student_feature_top, student_feature_bot]).permute(1,0)
                        student_features_selected = self.project_selected(student_features_selected)
                        distill_loss_positive = self.distill_loss(student_features_selected, teacher_features_selected, alpha = self.config.train["alpha"])
                        distill_loss_negative = -self.distill_loss(student_features_selected_negative, student_features_selected, alpha = self.config.train["beta"])
                        distill_loss = distill_loss_positive + distill_loss_negative

                    elif self.config.train['distill_mode'] == "image_level":
                        teacher_features = self.flat(teacher_features).mean(dim = 0)
                        student_features = self.project(self.flat(student_features).mean(dim = 0))
                        distill_loss =  self.distill_loss(student_features, teacher_features)
                total_loss = base_loss + distill_loss
                self.student_optimizer.zero_grad()
                total_loss.backward()
                self.student_optimizer.step()
                epoch_total_loss += total_loss.item()
                epoch_base_loss += base_loss.item()
                epoch_distill_loss += distill_loss.item()
                # if not (batch_num % 1):
                #     visualize_batch(self.student, student_image, student_target, self.config.networks["student_parameters"]["classes_names"][1], figsize = (7,7))
                #     self.student.train()

            self.student_scheduler.step()
            epoch_total_loss = epoch_total_loss / len(self.student_trainloader)
            epoch_base_loss = epoch_base_loss / len(self.student_trainloader)
            epoch_distill_loss = epoch_distill_loss / len(self.student_trainloader)
            current_metrics = self.test('student')
            print(
                "student_total_loss:", epoch_total_loss, 
                "student_base_loss:", epoch_base_loss, 
                "student_distill_loss:", epoch_distill_loss
                )
            print(current_metrics)
            # for logging in a single dictionary
            current_metrics["student_total_loss"] = epoch_total_loss
            current_metrics["student_base_loss"] = epoch_base_loss
            current_metrics["student_distill_loss"] = epoch_distill_loss
            wandb.log(current_metrics)
            self.save('student', path = self.config.networks["last_student_cp"])
            for k in current_metrics:
                if "mAP" in k:
                    current_metric = current_metrics[k]
            if current_metric >= best_metric:
                best_metric = current_metric
                print('saving best student checkpoint')
                self.save('student', path = self.config.networks["best_student_cp"])


    def test(self, mode, loader_mode='validation'):
        if mode == "student":
            if loader_mode=='training':
                dataloader = self.student_trainloader
            if loader_mode=='validation':
                dataloader = self.student_validloader
            if loader_mode=='testing':
                dataloader = self.student_testloader            
            network = self.student
            classes = self.config.networks["student_parameters"]["classes_names"]
        elif mode == "teacher":
            if loader_mode=='training':
                dataloader = self.teacher_trainloader
            if loader_mode=='validation':
                dataloader = self.teacher_validloader
            if loader_mode=='testing':
                dataloader = self.teacher_testloader            
            network = self.teacher
            classes = self.config.networks["teacher_parameters"]["classes_names"]
        print("Testing", mode, 'on', loader_mode, 'set')
        coco_metric = COCOMetric(
            classes=classes,
        )
        network.eval()
        with torch.no_grad():
            random_indices = [randint(0, len(dataloader)-1) for _ in range(20)]
            targets_all = []
            predictions_all = []
            for batch_num, batch in enumerate(tqdm(dataloader, unit="iter")):
                gc.collect()
                torch.cuda.empty_cache()
                images, targets = prepare_batch(batch, self.device)
                predictions = network(images)
                targets_all += targets
                predictions_all += predictions
                # if batch_num in random_indices:
                #     visualize_batch(network, images, targets, classes[1], figsize = (7,7))
            results_metric = matching_batch(
                iou_fn=box_iou,
                iou_thresholds=coco_metric.iou_thresholds,
                pred_boxes=[sample["boxes"].cpu().numpy() for sample in predictions_all],
                pred_classes=[sample["labels"].cpu().numpy() for sample in predictions_all],
                pred_scores=[sample["scores"].cpu().numpy() for sample in predictions_all],
                gt_boxes=[sample["boxes"].cpu().numpy() for sample in targets_all],
                gt_classes=[sample["labels"].cpu().numpy() for sample in targets_all],
            )
            logging.getLogger().disabled = True #disable logging warning for empty background
            metric_dict = coco_metric(results_metric)[0]
            logging.getLogger().disabled = False
            new_dict = {}
            for k in metric_dict:
                for c in classes[1:]: #remove background
                    if c in k: #not a background key
                        new_dict[mode+": "+k] = metric_dict[k]

        return new_dict


    def predict(self, path, mode = 'teacher'):
        if mode == 'student':
            network = self.student
            transforms = self.student_testloader.dataset.transform
        elif mode == 'teacher':
            network = self.teacher
            transforms = self.teacher_testloader.dataset.transform      
        network.eval()
        with torch.no_grad():
            predict_file = [{"image": path, 'boxes': torch.zeros((0,4)), 'labels':torch.zeros((0))}]
            predict_set = Dataset(
                data=predict_file, 
                transform=transforms
                )
            predict_loader = MonaiLoader(
                predict_set,
                batch_size = 1,
                num_workers = 0,
                pin_memory = False,
            )
            pred_boxes = []
            pred_scores = []
            for batch in predict_loader:
                batch["image"] = batch["image"].to(self.device)
                pred = network(batch["image"])
                pred_boxes += [sample["boxes"].cpu().numpy() for sample in pred],
                pred_scores += [sample["scores"].cpu().numpy() for sample in pred],

        pred_boxes = ([torch.tensor(pred_boxes[i][0]) for i in range(len(pred_boxes))])
        pred_scores = ([torch.tensor(pred_scores[i][0]) for i in range(len(pred_scores))])
        if pred_boxes == []:
            return torch.zeros((0,4)), torch.zeros((0))
        resized_boxes = []
        slice_shape = LoadImage()(path).shape
        for boxes, scores in zip(pred_boxes, pred_scores):
            scaling_factor_width = slice_shape[0] / self.config.transforms['size'][0]
            scaling_factor_height = slice_shape[1] / self.config.transforms['size'][1]
            boxes[:,0] = boxes[:,0] * scaling_factor_width
            boxes[:,1] = boxes[:,1] * scaling_factor_height
            boxes[:,2] = boxes[:,2] * scaling_factor_width
            boxes[:,3] = boxes[:,3] * scaling_factor_height
            resized_boxes.append(boxes)
        pred_boxes = resized_boxes
        pred_boxes = torch.stack(pred_boxes)
        pred_scores = torch.stack(pred_scores)
        return pred_boxes, pred_scores




    def predict_2dto3d(self, volume_path, mode = 'student', temp_path="temp/"):
        if mode == 'student':
            network = self.student
            transforms = self.student_testloader.dataset.transform
        elif mode == 'teacher':
            network = self.teacher
            transforms = self.teacher_testloader.dataset.transform      
        # Read volume
        loader = LoadImage()
        # img_volume = SimpleITK.ReadImage(volume_path)
        # img_volume_array = SimpleITK.GetArrayFromImage(img_volume)
        img_volume_array = loader(volume_path)
        slice_shape = img_volume_array[0].shape
        number_of_slices = img_volume_array.shape[0]
        # Create temporary folder to store 2d png files 
        if os.path.exists(temp_path) == False:
            os.mkdir(temp_path)
        # Write volume slices as 2d png files 
        for slice_number in range(4, number_of_slices-4):
            volume_slice = img_volume_array[slice_number, :, :]
            # if slice_number == int(14):
            #     plt.imshow(volume_slice)
            #     plt.show()
            # Delete extension from filename
            volume_file_name = os.path.splitext(volume_path)[0].split("/")[-1]
            volume_png_path = os.path.join(
                                    temp_path, 
                                    volume_file_name + "_" + str(slice_number)
                                    ) + ".png"
            volume_slice = np.asarray(volume_slice)
            volume_slice = ((volume_slice - volume_slice.min()) / (volume_slice.max() - volume_slice.min()) * 255).astype('uint8')
            cv2.imwrite(volume_png_path, volume_slice)
        # Predict slices individually then reconstruct 3D prediction
        network.eval()
        with torch.no_grad():
            volume_names = natsort.natsorted(os.listdir(temp_path))
            volume_paths = [os.path.join(temp_path, file_name) 
                            for file_name in volume_names]
            predict_files = [{"image": image_name, "boxes": np.zeros((0,4)), "labels": np.zeros((0))} 
                                for image_name in volume_paths]
            predict_set = Dataset(
                data=predict_files, 
                transform=transforms
                )
            predict_loader = MonaiLoader(
                predict_set,
                batch_size = 1,
                num_workers = 0,
                pin_memory = False,
            )
            pred_boxes = []
            pred_scores = []
            for batch in predict_loader:
                batch["image"] = batch["image"].to(self.device)
                pred = network(batch["image"])
                pred_boxes += [sample["boxes"].cpu().numpy() for sample in pred],
                pred_scores += [sample["scores"].cpu().numpy() for sample in pred],
        # Delete temporary folder
        shutil.rmtree(temp_path)
        pred_boxes = ([torch.tensor(pred_boxes[i][0]) for i in range(len(pred_boxes))])
        pred_scores = ([torch.tensor(pred_scores[i][0]) for i in range(len(pred_scores))])
        pred_boxes = [Boxes(coord) for coord in pred_boxes]
        final_boxes_vol ,final_scores_vol, final_slices_vol = NMS_volume(pred_boxes, pred_scores)
        if final_boxes_vol == []:
            return torch.zeros((0,6)), torch.zeros((0))
        resized_boxes = []
        for boxes, scores, slices in zip(final_boxes_vol ,final_scores_vol, final_slices_vol):
            scaling_factor_width = slice_shape[1] / self.config.transforms['size'][0]
            scaling_factor_height = slice_shape[0] / self.config.transforms['size'][1]
            boxes[0] = boxes[0] * scaling_factor_width
            boxes[1] = boxes[1] * scaling_factor_height
            boxes[2] = boxes[2] * scaling_factor_width
            boxes[3] = boxes[3] * scaling_factor_height
            resized_boxes.append(boxes)
        final_boxes_vol = resized_boxes
        final_boxes_3d = []
        for box, slice in zip(final_boxes_vol, final_slices_vol):
            zmin = max(0, slice - int(0.25 * number_of_slices)) # From DBTexs
            zmax = min(slice + int(0.25 * number_of_slices), number_of_slices-1)
            final_boxes_3d.append(torch.tensor([box[0], box[1], zmin, box[2], box[3], zmax]).int())
        final_boxes_vol = final_boxes_3d
        final_boxes_vol = torch.stack(final_boxes_vol)
        final_scores_vol = torch.stack(final_scores_vol)
        return final_boxes_vol, final_scores_vol