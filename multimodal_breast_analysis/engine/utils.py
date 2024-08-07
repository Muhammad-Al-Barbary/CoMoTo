import torch
import random
import numpy as np
import monai
import numpy as np
from typing import List, Tuple
import torch
from torch import device


def extract_critical_features(features, boxes, image_size, num_points = 9):
    """
    Extracts predetermined points from features maps foreground inside the target bounding boxes
    Args: 
        features: tensor: the extracted features maps of shape BxCxAxB
        boxes: list[tensor]: the target boxes of length B and shape N,4
        image_size: list: the original image size before downsampling
        num_points: int: the number of points to extract from the boxes, should 
                        be a value from 1, 4, 5, or 9 representing the center, edges, 
                        center + side midpoints, or all of them combined, respectively.
                        Note: choosing 1 point theoretically eliminate the effect of distillation 
                              due to the softmax activation of extracted feature points.
    """
    assert num_points in [1, 4, 5, 9]
    critical_features = torch.zeros((features.shape[0], features.shape[1], num_points), device = features.device)
    scaling_ratio = (features.shape[-1]) / (image_size[-1])
    for idx, (sample_features, sample_boxes) in enumerate(zip(features, boxes)):
        scaled_boxes = (sample_boxes * scaling_ratio).int()
        xmin = scaled_boxes[:, 0].clamp(max=sample_features.shape[-1]-1)
        ymin = scaled_boxes[:, 1].clamp(max=sample_features.shape[-2]-1)
        xmax = scaled_boxes[:, 2].clamp(max=sample_features.shape[-1]-1)
        ymax = scaled_boxes[:, 3].clamp(max=sample_features.shape[-2]-1)
        #TODO: Fix this
        if num_points == 1: # center only
            critical_features[idx,:,0] = sample_features[:, (ymin + ymax) // 2, (xmin + xmax) // 2].mean(-1)
        if num_points == 4: # corners
            critical_features[idx,:,0] = sample_features[:, ymin, xmin].mean(-1)
            critical_features[idx,:,1] = sample_features[:, ymin, xmax].mean(-1) 
            critical_features[idx,:,2] = sample_features[:, ymax, xmin].mean(-1)
            critical_features[idx,:,3] = sample_features[:, ymax, xmax].mean(-1)
        if num_points == 5: # center and midpoints
            critical_features[idx,:,0] = sample_features[:, (ymin + ymax) // 2, (xmin + xmax) // 2].mean(-1)
            critical_features[idx,:,1] = sample_features[:, (ymin + ymax) // 2, xmin].mean(-1)
            critical_features[idx,:,2] = sample_features[:, (ymin + ymax) // 2, xmax].mean(-1)
            critical_features[idx,:,3] = sample_features[:, ymin, (xmin + xmax) // 2].mean(-1)
            critical_features[idx,:,4] = sample_features[:, ymax, (xmin + xmax) // 2].mean(-1)
        if num_points == 9:  # center and midpoints and edges
            critical_features[idx,:,0] = sample_features[:, ymin, xmin].mean(-1) # average over all the boxes of the sample
            critical_features[idx,:,1] = sample_features[:, ymin, xmax].mean(-1) 
            critical_features[idx,:,2] = sample_features[:, ymax, xmin].mean(-1)
            critical_features[idx,:,3] = sample_features[:, ymax, xmax].mean(-1)
            critical_features[idx,:,4] = sample_features[:, (ymin + ymax) // 2, (xmin + xmax) // 2].mean(-1)
            critical_features[idx,:,5] = sample_features[:, (ymin + ymax) // 2, xmin].mean(-1)
            critical_features[idx,:,6] = sample_features[:, (ymin + ymax) // 2, xmax].mean(-1)
            critical_features[idx,:,7] = sample_features[:, ymin, (xmin + xmax) // 2].mean(-1)
            critical_features[idx,:,8] = sample_features[:, ymax, (xmin + xmax) // 2].mean(-1)
    return critical_features


def extract_noncritical_features(features, boxes, image_size, num_points = 9):
    """
    Extracts predetermined points from features maps background outside the target bounding boxes
    Args: 
        features: tensor: the extracted features maps of shape BxCxAxB
        boxes: list[tensor]: the target boxes of length B and shape N,4
        image_size: list: the original image size before downsampling
        num_points: int: the number of points to extract from the boxes, 
                         points are randomly sampled from the background.
    """
    features_selected_negative = []
    scaling_ratio = (features.shape[-1]) / (image_size[-1])
    for i in range(len(boxes)):
        boxes_mask = torch.ones((features.shape[-2], features.shape[-1]))
        for xmin,ymin,xmax,ymax in (boxes[i] * scaling_ratio).int():
            boxes_mask[ymin:ymax, xmin:xmax] = 0  
        positive_indices = torch.nonzero(boxes_mask == 1)
        shuffled_indices = positive_indices[torch.randperm(positive_indices.size(0))]
        sampled_indices = shuffled_indices[:min(num_points, shuffled_indices.size(0))].t()
        features_selected_negative.append(features[i, :, sampled_indices[0], sampled_indices[1]])
    features_selected_negative = torch.stack(features_selected_negative, dim = 0)
    return features_selected_negative


def prepare_batch(batch, device):
  image = [batch[i]["image"].to(device) for i in range(len(batch))]
  targets = [{"boxes":batch[i]["boxes"].to(device), "labels":batch[i]["labels"].to(device)} for i in range(len(batch))]
  return image, targets


def average_dicts(list_of_dicts):
  num_dicts = len(list_of_dicts)
  avg_dict = {}
  for d in list_of_dicts:
      for key, value in d.items():
          avg_dict[key] = avg_dict.get(key, 0) + value / num_dicts
  return avg_dict
  

def closest_index(lst, target):
    lst = np.array(lst)
    closest_index = np.abs(lst - target).argmin()
    return closest_index
  

def log_transforms(file_path, dataset_name):
    """
    Creates a string of the used transforms for documentation purposes
    Args:
        filepath: String: the path of the transforms file
        dataset_name: String: the used dataset
    """
    train_list, test_list = ["TRAIN:\n"],["TEST:\n"]
    with open(file_path, 'r') as file:
        lines = file.readlines()
    #get train and test sections from the file
    for line_idx in range(len(lines)):
      if "train_transforms" in lines[line_idx]:
        train_start = line_idx
      if "return" in lines[line_idx]:
        train_end = line_idx
        break
    for line_idx in range(train_end+1,len(lines)):
      if "test_transforms" in lines[line_idx]:
        test_start = line_idx
      if "return" in lines[line_idx]:
        test_end = line_idx
        break
    #print train transforms
    found_dataset = False
    for line_idx in range(train_start, train_end):
        if found_dataset and (":" in lines[line_idx] or "}" in lines[line_idx]):
            break
        if dataset_name in lines[line_idx]:
            found_dataset = True
        if found_dataset and not lines[line_idx].strip().startswith('#'):
            train_list.append(lines[line_idx])
    #print test transforms
    found_dataset = False
    for line_idx in range(test_start, test_end):
        if found_dataset and ":" in lines[line_idx]:
            break
        if dataset_name in lines[line_idx]:
            found_dataset = True
        if found_dataset and not lines[line_idx].strip().startswith('#'):
            test_list.append(lines[line_idx])
    train_list = ''.join(train_list)
    test_list = ''.join(test_list)
    return (train_list, test_list)


def set_seed(seed):
    """
    Sets seed for all randomized attributes of the packages and modules.
    Usually called before engine initialization. 
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    monai.utils.set_determinism(seed=seed)


def NMS_volume(pred_boxes_vol,pred_scores_vol):
    """
    Source: https://github.com/ICEBERG-VICOROB/DBT_phase2/blob/main/inference_DBT-NMS2.py
    """
    #TODO: Fix hardcoded values
    start_pred = 4 # 5start_pred:end to look for prediction not implemented
    end_pred = 4 # 5 end_pred to look for prediction not implemented (all slices are used)
    min_score = 0.001 # minimum score to keep prediction
    min_iou = 0.5 #0.5 #0.75 # minimum iou to look for overlappin boxes.
    min_score_s = 0.0 # minimum score for single predictions 
    depth_d = 4 #1 # division for depth 1= 100%
    length = len(pred_boxes_vol)
    final_boxes_vol = []
    final_scores_vol = []
    final_slices_vol = []
    depth = int(length/depth_d) # otherwise other lesions are still doble detected.
    # now get the maximum along the each slice depth
    for i in range(0,length):
        if (len(pred_boxes_vol[i])>0):
            s0 = max(i-depth,0)
            s1 = min(i+depth,length)
            # checking slice i against all other slices (doing a cat)           
            all_boxes = Boxes.cat(pred_boxes_vol[s0:s1])
            all_scores =  torch.squeeze(torch.cat([b for b in pred_scores_vol[s0:s1]], dim=0))
            # there is more than one annotation.
            if (all_scores.dim()>0):
                if (all_scores.shape[0] >0): # this is to solve bug found that some tensors were [0,1]
                    # check if there is overlap
                    iou_matrix = torch.squeeze(pairwise_iou(pred_boxes_vol[i], all_boxes))
                    j = 0
                    for box, score in zip(pred_boxes_vol[i],pred_scores_vol[i]):
                        if (iou_matrix.dim()>1):
                            idx_s = ((iou_matrix[j,:] > min_iou).nonzero())
                        else: 
                            idx_s = ((iou_matrix > min_iou).nonzero())
                        # condition 1: my score is larger or equal than all other overlapping boxes (idx_s)
                        if idx_s.numel() > 0 and score >= max(all_scores[idx_s]): # to fix bug idx_s is empty
                            if (score >= min_score):
                                final_boxes_vol.append(box)
                                final_scores_vol.append(score)
                                final_slices_vol.append(i+start_pred)
                        j +=1
            else: # there is only one annotation only check single threshold.
                for box, score in zip(pred_boxes_vol[i],pred_scores_vol[i]):
                    if (score >= min_score_s):
                        final_boxes_vol.append(box)
                        final_scores_vol.append(score)
                        final_slices_vol.append(i+start_pred)
    return final_boxes_vol ,final_scores_vol, final_slices_vol


# TODO: fix detectron2 import issue instead of copying code
class Boxes:
    """
    This structure stores a list of boxes as a Nx4 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx4. Each row is (x1, y1, x2, y2).
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, dtype=torch.float32, device=torch.device("cpu"))
        else:
            tensor = tensor.to(torch.float32)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()
        self.tensor = tensor


    def clone(self) -> "Boxes":
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        return Boxes(self.tensor.clone())


    def to(self, device: torch.device):
        # Boxes are assumed float32 and does not support to(dtype)
        return Boxes(self.tensor.to(device=device))


    def area(self) -> torch.Tensor:
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area


    def clip(self, box_size: Tuple[int, int]) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        h, w = box_size
        x1 = self.tensor[:, 0].clamp(min=0, max=w)
        y1 = self.tensor[:, 1].clamp(min=0, max=h)
        x2 = self.tensor[:, 2].clamp(min=0, max=w)
        y2 = self.tensor[:, 3].clamp(min=0, max=h)
        self.tensor = torch.stack((x1, y1, x2, y2), dim=-1)


    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep


    def __getitem__(self, item) -> "Boxes":
        """
        Args:
            item: int, slice, or a BoolTensor

        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.

        The following usage are allowed:

        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Boxes(b)


    def __len__(self) -> int:
        return self.tensor.shape[0]


    def __repr__(self) -> str:
        return "Boxes(" + str(self.tensor) + ")"


    def inside_box(self, box_size: Tuple[int, int], boundary_threshold: int = 0) -> torch.Tensor:
        """
        Args:
            box_size (height, width): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        height, width = box_size
        inds_inside = (
            (self.tensor[..., 0] >= -boundary_threshold)
            & (self.tensor[..., 1] >= -boundary_threshold)
            & (self.tensor[..., 2] < width + boundary_threshold)
            & (self.tensor[..., 3] < height + boundary_threshold)
        )
        return inds_inside


    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The box centers in a Nx2 array of (x, y).
        """
        return (self.tensor[:, :2] + self.tensor[:, 2:]) / 2


    def scale(self, scale_x: float, scale_y: float) -> None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y


    @classmethod
    def cat(cls, boxes_list: List["Boxes"]) -> "Boxes":
        """
        Concatenates a list of Boxes into a single Boxes

        Arguments:
            boxes_list (list[Boxes])

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(box, Boxes) for box in boxes_list])
        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes


    @property
    def device(self) -> device:
        return self.tensor.device
    

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self):
        """
        Yield a box as a Tensor of shape (4,) at a time.
        """
        yield from self.tensor


def pairwise_intersection(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the intersection area between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax)

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: intersection, sized [N,M].
    """
    boxes1, boxes2 = boxes1.tensor, boxes2.tensor
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]
    width_height.clamp_(min=0)  # [N,M,2]
    intersection = width_height.prod(dim=2)  # [N,M]
    return intersection


def pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M, compute the IoU
    (intersection over union) between **all** N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [M]
    inter = pairwise_intersection(boxes1, boxes2)
    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou