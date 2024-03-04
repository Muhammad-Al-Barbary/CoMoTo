import torch
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
import warnings
from random import randint
import numpy as np
import pydicom

def visualize_batch(model, images, targets, class_name, figsize=(10,10), threshold = 0.25):
    warnings.filterwarnings("ignore") #disable warning for empty boxes
    model.eval()
    predictions = model(images)
    random_idx = randint(0, len(images) - 1)
    pred = predictions[random_idx]
    thresholded_boxes = []
    thresholded_scores = []
    thresholded_labels = []
    for i in range(len(pred["boxes"])):
        if pred["scores"][i] >= threshold:
            thresholded_boxes.append(pred["boxes"][i])
            thresholded_scores.append(pred["scores"][i])
            thresholded_labels.append(pred["labels"][i])
    if len(thresholded_labels) > 0:
        pred["scores"] = torch.stack(thresholded_scores)
        pred["labels"] = torch.stack(thresholded_labels)
        pred["boxes"] = torch.stack(thresholded_boxes)
    else: 
        pred["scores"] = torch.zeros((0))
        pred["labels"] = torch.zeros((0))
        pred["boxes"] = torch.zeros((0,4))
    image = images[random_idx]
    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    pred_labels = [f"{class_name}: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()  
    target_labels = [f"{class_name}" for label in targets[random_idx]["labels"]]
    target_image = draw_bounding_boxes(image, targets[random_idx]["boxes"], target_labels, colors="red", width = int(image.shape[1]/250));
    plt.figure(figsize=figsize)
    plt.subplot(1,2,1)
    plt.title("target")
    plt.axis("off")
    plt.imshow(target_image.permute(1,2,0).cpu())
    plt.subplot(1,2,2)
    plt.title("prediction")
    plt.axis("off")
    print(f"boxes:{pred_boxes}, labels:{pred_labels}")
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red", width = int(image.shape[1]/250));
    plt.imshow(output_image.permute(1, 2, 0).cpu())
    plt.show()


def visualize_dataset_sample(sample, figsize = (10,10)):
    if sample["image"].split(".")[-1] in ["png", "jpg", "jpeg"]:
        image = plt.imread(sample["image"])
    elif sample["image"].split(".")[-1] == "dcm":
        image =  pydicom.dcmread(sample["image"]).pixel_array
    image = (255.0 * (image - image.min()) / (image.max() - image.min())).astype(np.uint8)
    image = torch.tensor(image).unsqueeze(dim=0)
    labels = [f"{label}" for label in sample["labels"]]
    boxes = torch.tensor(sample["boxes"]).long()
    output_image = draw_bounding_boxes(image, boxes, labels, colors="red", width = int(image.shape[1]/250));
    plt.figure(figsize=figsize)
    plt.imshow(output_image.permute(1,2,0))
    plt.title("target")
    plt.axis("off")
    plt.show()