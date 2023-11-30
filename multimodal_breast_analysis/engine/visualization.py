import os
from torchvision.io import read_image
import torch
from torchvision.ops.boxes import masks_to_boxes
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes

def visualize_batch(model, images, targets, class_name):
    model.eval()
    predictions = model(images)
    pred = predictions[0]
    image = images[0]
    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    pred_labels = [f"{class_name}: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()  
    target_labels = [f"{class_name}: {1:.3f}" for label in targets[0]["labels"]]
    target_image = draw_bounding_boxes(image, targets[0]["boxes"], target_labels, colors="red");
    plt.subplot(1,2,1)
    plt.title("target")
    plt.axis("off")
    plt.imshow(target_image.permute(1,2,0).cpu())
    plt.subplot(1,2,2)
    plt.title("prediction")
    plt.axis("off")
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red");
    plt.imshow(output_image.permute(1, 2, 0).cpu())
    plt.show()