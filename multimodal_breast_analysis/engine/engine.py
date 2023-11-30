from multimodal_breast_analysis.models.faster_rcnn import faster_rcnn
from multimodal_breast_analysis.models.unet import UNet as unet
from multimodal_breast_analysis.configs.configs import config
from multimodal_breast_analysis.data.dataloader import DataLoader
from multimodal_breast_analysis.data.transforms import train_transforms, test_transforms
from multimodal_breast_analysis.data.datasets import penn_fudan
from multimodal_breast_analysis.engine.utils import prepare_batch, log_transforms, average_dicts
from multimodal_breast_analysis.engine.visualization import visualize_batch

import wandb
import logging
from tqdm import tqdm
import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CyclicLR
from torch.nn import KLDivLoss, Flatten, Linear
from torch.nn.functional import softmax, log_softmax
from monai.data.box_utils import box_iou
from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch

class Engine:
    def __init__(self):
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
                            data=self._get_data(config.data["student_name"])(config.data["student_path"]),
                            test_split=config.data['test_split'],
                            seed=config.seed
                            )
        teacher_loader = DataLoader(
                            data=self._get_data(config.data["teacher_name"])(config.data["teacher_path"]),
                            test_split=config.data['test_split'],
                            seed=config.seed
                            )
        self.student_trainloader = student_loader.trainloader(
                                        train_transforms(config.data["student_name"]), 
                                        batch_size=config.data["batch_size"], 
                                        shuffle=config.data["shuffle"]
                                        )
        self.student_testloader = student_loader.testloader(
                                        test_transforms(config.data["student_name"]), 
                                        batch_size=config.data["batch_size"], 
                                        shuffle=False
                                        )
        self.teacher_trainloader = teacher_loader.trainloader(
                                        train_transforms(config.data["teacher_name"]), 
                                        batch_size=config.data["batch_size"], 
                                        shuffle=config.data["shuffle"]
                                        )
        self.teacher_testloader = teacher_loader.testloader(
                                        test_transforms(config.data["teacher_name"]), 
                                        batch_size=config.data["batch_size"], 
                                        shuffle=False
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
             } #
        return data[dataset_name]

    def _get_model(self, name, parameters):
        models = {
            "faster_rcnn": faster_rcnn,
            "unet": unet
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
                "optimizer": self.teacher_optimizer.state_dict(),
                }
        elif mode == 'student':
            if path is None:
                path = config.networks["student_cp"]
            checkpoint = {
                "network": self.student.state_dict(),
                "optimizer": self.student_optimizer.state_dict(),
                }            
        torch.save(checkpoint, path)


    def load(self, mode, path=None):
        if mode == "teacher":
            if path is None:
                path = config.networks["teacher_cp"]
            checkpoint = torch.load(path)
            self.teacher.load_state_dict(checkpoint['network'])
            self.teacher_optimizer.load_state_dict(checkpoint['optimizer'])
        elif mode == "student":
            if path is None:
                path = config.networks["student_cp"]
            checkpoint = torch.load(path)
            self.student.load_state_dict(checkpoint['network'])
            self.student_optimizer.load_state_dict(checkpoint['optimizer'])


    def warmup(self):
        warmup_epochs = config.train["warmup_epochs"]
        best_metric = 0
        for epoch in range(warmup_epochs):
            print(f"\nWarmup Epoch {epoch+1}/{warmup_epochs}\n-------------------------------")
            epoch_loss = 0
            for batch in tqdm(self.teacher_trainloader, unit="iter"):
                self.teacher.train()
                image, target = prepare_batch(batch, self.device)
                loss = self.teacher(
                            image,
                            target
                            )
                loss = sum(sample_loss for sample_loss in loss.values())
                self.teacher_optimizer.zero_grad()
                loss.backward()
                self.teacher_optimizer.step()
                epoch_loss += loss.item()
                # print("   Loss:", loss.item())
                # wandb.log({"Warmup Loss": loss.item()})
                # visualize_batch(self.teacher, image, target, "pedestrian")
            epoch_loss = epoch_loss / len(self.teacher_trainloader)
            current_metrics = self.test('teacher')
            print("teacher_loss", epoch_loss)   
            print(current_metrics)
            # for logging in a single dictionary
            current_metrics["teacher_loss"] = epoch_loss
            wandb.log(current_metrics)
            if True:#current_metric >= best_metric:
                #best_metric = current_metric
                print('saving teacher checkpoint')
                self.save('teacher')


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


    def train(self):
        self._instantiate_kd()
        epochs = config.train["epochs"]
        distill_epoch = config.train["distill_epoch"]
        distill_coeff = config.train["distill_coeff"]
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
                    teacher_image, _ = prepare_batch(teacher_batch, self.device)
                    self.teacher(teacher_image)
                    teacher_features = self.features['teacher']
                    student_features = self.features['student']
                    teacher_features = self.flat(teacher_features).mean(dim = 0)
                    student_features = self.project(self.flat(student_features).mean(dim = 0))
                    distill_loss = distill_coeff * self.distill_loss(student_features, teacher_features)
                total_loss = base_loss + distill_loss
                self.student_optimizer.zero_grad()
                total_loss.backward()
                self.student_optimizer.step()
                epoch_total_loss += total_loss.item()
                epoch_base_loss += base_loss.item()
                epoch_distill_loss += distill_loss.item()
                # print(
                #     "    Total Loss:", total_loss.item(), 
                #     "Base:", base_loss.item(), 
                #     "Distill:", distill_loss.item()
                #     )
                # wandb.log({
                #     "student total training loss": total_loss.item(),
                #     "student base training loss": base_loss.item(),
                #     "student distill training loss": distill_loss.item(), 
                # })
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
            if True:#current_metric >= best_metric:
                #best_metric = current_metric
                print('saving student checkpoint')
                self.save('student')


    def test(self, mode):
        if mode == "student":
            dataloader = self.student_testloader
            network = self.student
            classes = config.networks["student_parameters"]["classes_names"]
        elif mode == "teacher":
            dataloader = self.teacher_testloader
            network = self.teacher
            classes = config.networks["teacher_parameters"]["classes_names"]
        print("Testing", mode)
        coco_metric = COCOMetric(
            classes=classes,
        )
        network.eval()
        with torch.no_grad():
          all_dicts = []
          for batch in tqdm(dataloader, unit="iter"):
            images, targets = prepare_batch(batch, self.device)
            predictions = network(images)
            results_metric = matching_batch(
                iou_fn=box_iou,
                iou_thresholds=coco_metric.iou_thresholds,
                pred_boxes=[sample["boxes"].cpu().numpy() for sample in predictions],
                pred_classes=[sample["labels"].cpu().numpy() for sample in predictions],
                pred_scores=[sample["scores"].cpu().numpy() for sample in predictions],
                gt_boxes=[sample["boxes"].cpu().numpy() for sample in targets],
                gt_classes=[sample["labels"].cpu().numpy() for sample in targets],
            )
            logging.getLogger().disabled = True #disable logging warning for empty background
            metric_dict = coco_metric(results_metric)[0]
            logging.getLogger().disabled = False
            new_dict = {}
            for k in metric_dict:
              for c in classes[1:]: #remove background
                if c in k: #not a background key
                  new_dict[mode+": "+k] = metric_dict[k]
            all_dicts.append(new_dict)
          visualize_batch(network, images, targets, classes[1])
        return average_dicts(all_dicts)


    def predict(self):
        pass