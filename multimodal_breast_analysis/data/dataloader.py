from monai.data import Dataset, DataLoader as MonaiLoader
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, data, test_split, seed):
        self.train_data, self.test_data = train_test_split(data, test_size=test_split, random_state=seed)
         
    def trainloader(self, transforms, batch_size, shuffle):
        dataset = Dataset(self.train_data, transform = transforms)
        dataloader = MonaiLoader(dataset, batch_size = batch_size, shuffle = shuffle, collate_fn=lambda batch: batch)
        return dataloader
    
    def testloader(self, transforms, batch_size, shuffle):
        dataset = Dataset(self.test_data, transform = transforms)
        dataloader = MonaiLoader(dataset, batch_size = batch_size, shuffle = shuffle, collate_fn=lambda batch: batch)
        return dataloader



# import os
# from torchvision.io import read_image
# import torch
# from torchvision.ops.boxes import masks_to_boxes
# def prepare_batch(batch):
#   image = [batch[i]["image"].to("cuda") for i in range(len(batch))]
#   targets = [{"boxes":batch[i]["boxes"].to("cuda"), "labels":batch[i]["labels"].to("cuda")} for i in range(len(batch))]
#   return image, targets
# for batch in dataloader:
#     image, target = prepare_batch(batch)
    # loss = model(image, target)
    # model.eval()
    # predictions = model(image)
    # import matplotlib.pyplot as plt
    # from torchvision.utils import draw_bounding_boxes
    # pred = predictions[0]
    # image = image[0]
    # image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    # pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    # pred_boxes = pred["boxes"].long()  
    # target_labels = [f"pedestrian: {1:.3f}" for label in target[0]["labels"]]
    # target_image = draw_bounding_boxes(image, target[0]["boxes"], target_labels, colors="red");
    # plt.imshow(target_image.permute(1,2,0))
    # plt.show()
    # output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red");
    # plt.imshow(output_image.permute(1, 2, 0))
    # plt.show()