from multimodal_breast_analysis.models.faster_rcnn import faster_rcnn, faster_rcnn_fpn
from multimodal_breast_analysis.models.unet import UNet as unet
from multimodal_breast_analysis.configs.configs import config
from multimodal_breast_analysis.data.dataloader import DataLoader
from multimodal_breast_analysis.data.transforms import train_transforms, test_transforms
from multimodal_breast_analysis.data.datasets import penn_fudan, omidb, omidb_downsampled, dbt_2d
from multimodal_breast_analysis.engine.utils import prepare_batch, log_transforms, average_dicts, set_seed
from multimodal_breast_analysis.engine.visualization import visualize_batch

import wandb
import logging
from tqdm import tqdm
import torch
import torchvision
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CyclicLR
from torch.nn import KLDivLoss, Flatten, Linear
from torch.nn.functional import softmax, log_softmax
from monai.data.box_utils import box_iou
from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch
import gc
from random import randint

class Engine:
    def __init__(self):
        set_seed(config.seed)
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.student = self._get_model(
            config.networks["student"],
            config.networks["student_parameters"]
            ).to(self.device)
        self.teacher = self._get_model(
            config.networks["teacher"], 
            config.networks["teacher_parameters"]
            ).to(self.device)    
        self.student_optimizer = self._get_optimizer(
            config.train["student_optimizer"]
        )(self.student.parameters(),**config.train["student_optimizer_parameters"])
        self.teacher_optimizer = self._get_optimizer(
            config.train["teacher_optimizer"]
        )(self.teacher.parameters(),**config.train["teacher_optimizer_parameters"])
        self.student_scheduler = self._get_scheduler(
            config.train["student_scheduler"],
        )(self.student_optimizer,**config.train["student_scheduler_parameters"])
        self.teacher_scheduler = self._get_scheduler(
            config.train["teacher_scheduler"],
        )(self.teacher_optimizer,**config.train["teacher_scheduler_parameters"])


        student_loader = DataLoader(
                            data=self._get_data(config.data["student_name"])(*config.data["student_args"]),
                            valid_split=config.data['valid_split'],
                            test_split=config.data['test_split'],
                            seed=config.seed
                            )
        teacher_loader = DataLoader(
                            data=self._get_data(config.data["teacher_name"])(*config.data["teacher_args"]),
                            valid_split=config.data['valid_split'],
                            test_split=config.data['test_split'],
                            seed=config.seed
                            )
        self.student_trainloader = student_loader.trainloader(
                                       train_transforms(config.data["student_name"]), 
                                        batch_size=config.data["batch_size"], 
                                        shuffle=config.data["shuffle"],
                                        train_ratio=config.data["train_ratio"],
                                        )
        self.student_validloader = student_loader.validloader(
                                        test_transforms(config.data["student_name"]), 
                                        batch_size=config.data["batch_size"], 
                                        )
        self.student_testloader = student_loader.testloader(
                                        test_transforms(config.data["student_name"]), 
                                        batch_size=config.data["batch_size"], 
                                        )

        self.teacher_trainloader = teacher_loader.trainloader(
                                        train_transforms(config.data["teacher_name"]), 
                                        batch_size=config.data["batch_size"], 
                                        shuffle=config.data["shuffle"]
                                        )
        self.teacher_validloader = teacher_loader.validloader(
                                        test_transforms(config.data["teacher_name"]), 
                                        batch_size=config.data["batch_size"], 
                                        )
        self.teacher_testloader = teacher_loader.testloader(
                                        test_transforms(config.data["teacher_name"]), 
                                        batch_size=config.data["batch_size"], 
                                        )

        student_train_logs, student_test_logs = log_transforms(
                                                                "multimodal_breast_analysis/data/transforms.py", 
                                                                config.data["student_name"]
                                                                )
        teacher_train_logs, teacher_test_logs = log_transforms(
                                                                "multimodal_breast_analysis/data/transforms.py", 
                                                                config.data["teacher_name"]
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
                path = config.networks["teacher_cp"]
            checkpoint = {
                "network": self.teacher.state_dict(),
                # "optimizer": self.teacher_optimizer.state_dict(),
                }
        elif mode == 'student':
            if path is None:
                path = config.networks["student_cp"]
            checkpoint = {
                "network": self.student.state_dict(),
                # "optimizer": self.student_optimizer.state_dict(),
                }            
        torch.save(checkpoint, path)


    def load(self, mode, path=None):
        if mode == "teacher":
            if path is None:
                path = config.networks["teacher_cp"]
            checkpoint = torch.load(path)
            self.teacher.load_state_dict(checkpoint['network'])
            # self.teacher_optimizer.load_state_dict(checkpoint['optimizer'])
        elif mode == "student":
            if path is None:
                path = config.networks["student_cp"]
            checkpoint = torch.load(path)
            self.student.load_state_dict(checkpoint['network'])
            # self.student_optimizer.load_state_dict(checkpoint['optimizer'])


    def warmup(self):
        # print('initial metrics:', self.test('teacher'))
        CROSS_ALIGN = False # not available yet
        if CROSS_ALIGN:
            from torch.nn.functional import cosine_similarity
            from torch import sigmoid
            def similarity_loss(tensor):
                reshaped_tensor = tensor.view(tensor.size(0), -1)                
                similarity_matrix = cosine_similarity(reshaped_tensor.unsqueeze(1), reshaped_tensor.unsqueeze(0), dim=2)
                loss = (torch.sum(similarity_matrix) - torch.trace(similarity_matrix)) / (tensor.size(0) * (tensor.size(0) - 1))
                return sigmoid(-loss)

            self.features = {}
            def get_features(name):
                def hook(model, input, output):
                    self.features[name] = output
                return hook
            self.teacher.backbone.register_forward_hook(get_features('teacher'))

        warmup_epochs = config.train["warmup_epochs"]
        best_metric = 0
        for epoch in range(warmup_epochs):
            print(f"\nWarmup Epoch {epoch+1}/{warmup_epochs}\n-------------------------------")
            epoch_loss = 0
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

##############################################Ignore########################################################################################
                if CROSS_ALIGN: #not available yet
                    teacher_features = self.features['teacher']
                    teacher_ratio = teacher_features.shape[-1] / image[0].shape[-1]
                    teacher_boxes = [(target[i]['boxes'] * teacher_ratio).round().int() for i in range(len(target))]
                    for i in range(len(teacher_boxes)):
                        teacher_boxes[i][:,0] = teacher_boxes[i][:,0].clamp(max = teacher_features.shape[-1]-1)
                        teacher_boxes[i][:,1] = teacher_boxes[i][:,1].clamp(max = teacher_features.shape[-2]-1)
                        teacher_boxes[i][:,2] = teacher_boxes[i][:,2].clamp(max = teacher_features.shape[-1]-1)
                        teacher_boxes[i][:,3] = teacher_boxes[i][:,3].clamp(max = teacher_features.shape[-2]-1)
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
                    
                    # teacher_feature_botleft, teacher_feature_botright, teacher_feature_topleft, teacher_feature_topright
                    # teacher_feature_center, teacher_feature_left, teacher_feature_right, teacher_feature_top, teacher_feature_bot
                    teacher_feature_botleft = torch.stack(teacher_feature_botleft)
                    teacher_feature_botright = torch.stack(teacher_feature_botright)
                    teacher_feature_topleft = torch.stack(teacher_feature_topleft)
                    teacher_feature_topright = torch.stack(teacher_feature_topright)
                    teacher_feature_center = torch.stack(teacher_feature_center)
                    teacher_feature_left = torch.stack(teacher_feature_left)
                    teacher_feature_right = torch.stack(teacher_feature_right)
                    teacher_feature_top = torch.stack(teacher_feature_top)
                    teacher_feature_bot = torch.stack(teacher_feature_bot)


                    teacher_features_selected = torch.stack(
                        [teacher_feature_botleft, teacher_feature_botright, teacher_feature_topleft, teacher_feature_topright, teacher_feature_center,
                          teacher_feature_left, teacher_feature_right, teacher_feature_top, teacher_feature_bot],
                          dim = -1
                          )#.permute(1,0)
                    
                    sim_loss = similarity_loss(teacher_features_selected)
                    # print('detection loss:', loss, 'similarity loss:', sim_loss)
                    loss = loss + sim_loss
                    # print('total loss:', loss)
                    # print(teacher_features_selected.shape)    
######################################################################################################################################



                self.teacher_optimizer.zero_grad()
                loss.backward()
                self.teacher_optimizer.step()
                epoch_loss += loss.item()
                # print("   Loss:", loss.item())
                # wandb.log({"Warmup Loss": loss.item()})
                # if not (batch_num % 30):
                #     visualize_batch(self.teacher, image, target, config.networks["teacher_parameters"]["classes_names"][1], figsize = (7,7))
            self.teacher_scheduler.step()
            epoch_loss = epoch_loss / len(self.teacher_trainloader)
            current_metrics = self.test('teacher')
            print("teacher_loss", epoch_loss)   
            print(current_metrics)
            # for logging in a single dictionary
            current_metrics["teacher_loss"] = epoch_loss
            wandb.log(current_metrics)
            self.save('teacher', path = config.networks["last_teacher_cp"])
            for k in current_metrics:
                if "mAP" in k:
                    current_metric = current_metrics[k]
            if current_metric >= best_metric:
                best_metric = current_metric
                print('saving best teacher checkpoint')
                self.save('teacher', path = config.networks["best_teacher_cp"])


    def distill_loss(self, student_outputs, teacher_outputs):
        T, alpha = config.train["temperature"], config.train["alpha"]
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
            student_batch = next(iter(self.student_trainloader))
            teacher_batch = next(iter(self.teacher_trainloader))
            student_image, student_target = prepare_batch(student_batch, self.device)
            teacher_image, teacher_target = prepare_batch(teacher_batch, self.device)
            self.student(student_image, student_target)
            self.teacher(teacher_image, teacher_target)
            student_features = self.features['student']
            teacher_features = self.features['teacher']
            # ratio = int(teacher_features.shape[1] / student_features.shape[1])
            # self.pooler = torch.nn.MaxPool1d(ratio, stride=ratio, padding=0)
            # self.pooler_selected = torch.nn.MaxPool2d(3, stride=1, padding=1)
            self.flat = Flatten(start_dim=2).to(self.device)
            self.project = Linear(
                                torch.prod(torch.tensor(student_features.shape[2:])),
                                torch.prod(torch.tensor(teacher_features.shape[2:]))
                                ).to(self.device)
            self.student_optimizer.add_param_group({'params':self.project.parameters()})

            self.project_selected = Linear(9, 9).to(self.device)
            self.student_optimizer.add_param_group({'params':self.project_selected.parameters()})

            # self.project_selected_sim = Linear(2048, 2048).to(self.device)
            # self.student_optimizer.add_param_group({'params':self.project_selected_sim.parameters()})

    def train(self):
        self._instantiate_kd()
        epochs = config.train["epochs"]
        distill_epoch = config.train["distill_epoch"]
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
            for student_batch in tqdm(self.student_trainloader, unit="iter"):
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
                        # if (len(teacher_batch['image']) != len(student_batch['image'])):
                        #     continue
                    teacher_image, teacher_target = prepare_batch(teacher_batch, self.device)
                    with torch.no_grad():
                        self.teacher(teacher_image)
                    teacher_features = self.features['teacher']
                    student_features = self.features['student']
                    
                    # student_features = self.pooler_selected(student_features)
                    # teacher_features = self.pooler_selected(teacher_features)

                    ##################### Lesion-specific KD ####################################################
                    student_ratio = student_features.shape[-1] / student_image[0].shape[-1]
                    teacher_ratio = teacher_features.shape[-1] / teacher_image[0].shape[-1]
                    student_boxes = [(student_target[i]['boxes'] * student_ratio).round().int() for i in range(len(student_target))]
                    teacher_boxes = [(teacher_target[i]['boxes'] * teacher_ratio).round().int() for i in range(len(teacher_target))]
                    for i in range(len(student_boxes)):
                        student_boxes[i][:,0] = student_boxes[i][:,0].clamp(max = student_features.shape[-1]-1)
                        student_boxes[i][:,1] = student_boxes[i][:,1].clamp(max = student_features.shape[-2]-1)
                        student_boxes[i][:,2] = student_boxes[i][:,2].clamp(max = student_features.shape[-1]-1)
                        student_boxes[i][:,3] = student_boxes[i][:,3].clamp(max = student_features.shape[-2]-1)
                    for i in range(len(teacher_boxes)):
                        teacher_boxes[i][:,0] = teacher_boxes[i][:,0].clamp(max = teacher_features.shape[-1]-1)
                        teacher_boxes[i][:,1] = teacher_boxes[i][:,1].clamp(max = teacher_features.shape[-2]-1)
                        teacher_boxes[i][:,2] = teacher_boxes[i][:,2].clamp(max = teacher_features.shape[-1]-1)
                        teacher_boxes[i][:,3] = teacher_boxes[i][:,3].clamp(max = teacher_features.shape[-2]-1)
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
                    selected_ratio = teacher_features[0][0].shape[0]/9
                    #####################################################################################################



                    teacher_features = self.flat(teacher_features).mean(dim = 0)
                    student_features = self.project(self.flat(student_features).mean(dim = 0))
                    # student_features_selected_sim = self.project_selected_sim(torch.cosine_similarity(student_features_selected.unsqueeze(1), student_features_selected.unsqueeze(0), dim=-1))
                    # teacher_features_selected_sim = torch.cosine_similarity(teacher_features_selected.unsqueeze(1), teacher_features_selected.unsqueeze(0), dim=-1)
                    # student_features_selected_sim = torch.cosine_similarity(student_features_selected.permute(1,0).unsqueeze(1), student_features_selected.permute(1,0).unsqueeze(0), dim=-1)
                    # teacher_features_selected_sim = torch.cosine_similarity(teacher_features_selected.permute(1,0).unsqueeze(1), teacher_features_selected.permute(1,0).unsqueeze(0), dim=-1)
                    # # teacher_features = self.pooler(teacher_features.permute(1,0)).permute(1,0)
                    # teacher_features = teacher_features[torch.linspace(0, teacher_features.shape[0]-1, student_features.shape[0]).long()]
                    
                    distill_loss =  self.distill_loss(student_features, teacher_features)#  / selected_ratio
                    # distill_loss = self.distill_loss(student_features_selected, teacher_features_selected)
                    # distill_loss = self.distill_loss(student_features_selected_sim, teacher_features_selected_sim)


                total_loss = base_loss + distill_loss
                self.teacher_optimizer.zero_grad()
                total_loss.backward()
                self.student_optimizer.step()
                epoch_total_loss += total_loss.item()
                epoch_base_loss += base_loss.item()
                epoch_distill_loss += distill_loss.item()
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
            self.save('student', path = config.networks["last_student_cp"])
            for k in current_metrics:
                if "mAP" in k:
                    current_metric = current_metrics[k]
            if current_metric >= best_metric:
                best_metric = current_metric
                print('saving best student checkpoint')
                self.save('student', path = config.networks["best_student_cp"])


    def test(self, mode, loader_mode='validation'):
        if mode == "student":
            if loader_mode=='training':
                dataloader = self.student_trainloader
            if loader_mode=='validation':
                dataloader = self.student_validloader
            if loader_mode=='testing':
                dataloader = self.student_testloader            
            network = self.student
            classes = config.networks["student_parameters"]["classes_names"]
        elif mode == "teacher":
            if loader_mode=='training':
                dataloader = self.teacher_trainloader
            if loader_mode=='validation':
                dataloader = self.teacher_validloader
            if loader_mode=='testing':
                dataloader = self.teacher_testloader            
            network = self.teacher
            classes = config.networks["teacher_parameters"]["classes_names"]
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


    def predict(self):
        pass #TODO