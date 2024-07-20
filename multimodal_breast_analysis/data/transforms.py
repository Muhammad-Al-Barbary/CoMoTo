import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Rotate90d,
    Flipd,
    Resized,
    RandFlipd,
)
from monai.apps.detection.transforms.dictionary import (
    BoxToMaskd,
    MaskToBoxd,
)


def train_transforms(dataset_name, configs):
    """
    Retrieves a Compose of training transforms based on the dataset and the configurations
    Args: 
        dataset_name: String: the name of the desired dataset as defined in the transforms dict inside this function
        configs: dict: a configuration dict containing transforms parameters
    """
    resize = configs['size']
    image_key = "image"
    box_key = "boxes"
    label_key = "labels"
    transforms = {
        "omidb": Compose([
            LoadImaged(keys=[image_key], meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys=[image_key]),
            EnsureTyped(keys=[label_key,box_key], dtype=torch.long),
            EnsureTyped(keys=[image_key], dtype=torch.float16),
            BoxToMaskd(
                box_keys=[box_key],
                label_keys=[label_key],
                box_mask_keys=["box_mask"],
                box_ref_image_keys=image_key,
                min_fg_label=0,
                ellipse_mask=True,
            ),
            Resized(
                keys=[image_key, "box_mask"],
                spatial_size=resize,
                mode=('bilinear','nearest')
            ),
            RandFlipd(keys=[image_key, "box_mask"], prob = 0.5, spatial_axis = 0),
            MaskToBoxd(
                box_keys=[box_key],
                label_keys=[label_key],
                box_mask_keys=["box_mask"],
                min_fg_label=0,
            ),
            Rotate90d(keys=image_key, k=3), # to fix loader orientation convention
            Flipd(image_key, 1), # to fix loader orientation convention
            EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
            EnsureTyped(keys=[label_key], dtype=int),
        ]),
        
        "dbt": Compose([
            LoadImaged(keys=[image_key], meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys=[image_key]),
            EnsureTyped(keys=[label_key,box_key], dtype=torch.long),
            EnsureTyped(keys=[image_key], dtype=torch.float16),
            BoxToMaskd(
                box_keys=[box_key],
                label_keys=[label_key],
                box_mask_keys=["box_mask"],
                box_ref_image_keys=image_key,
                min_fg_label=0,
                ellipse_mask=True,
            ),
            Resized(
                keys=[image_key, "box_mask"],
                spatial_size=resize,
                mode=('bilinear','nearest')
            ),
            RandFlipd(keys=[image_key, "box_mask"], prob = 0.5, spatial_axis = 0),
            MaskToBoxd(
                box_keys=[box_key],
                label_keys=[label_key],
                box_mask_keys=["box_mask"],
                min_fg_label=0,
            ),
            Rotate90d(keys=image_key, k=3),
            Flipd(image_key, 1),
            EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
            EnsureTyped(keys=[label_key], dtype=int),
        ])
    }
    return transforms[dataset_name]


def test_transforms(dataset_name, configs):
    """
    Retrieves a Compose of evaluation transforms based on the dataset and the configurations
    Args: 
        dataset_name: String: the name of the desired dataset as defined in the transforms dict inside this function
        configs: dict: a configuration dict containing transforms parameters
    """
    resize = configs['size']
    image_key = "image"
    box_key = "boxes"
    label_key = "labels"
    transforms = {
        "omidb": Compose([
            LoadImaged(keys=[image_key], meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys=[image_key]),
            EnsureTyped(keys=[label_key,box_key], dtype=torch.long),
            EnsureTyped(keys=[image_key], dtype=torch.float16),
            BoxToMaskd(
                box_keys=[box_key],
                label_keys=[label_key],
                box_mask_keys=["box_mask"],
                box_ref_image_keys=image_key,
                min_fg_label=0,
                ellipse_mask=True,
            ),
            Resized(
                keys=[image_key, "box_mask"],
                spatial_size=resize,
                mode=('bilinear','nearest')
            ),
            MaskToBoxd(
                box_keys=[box_key],
                label_keys=[label_key],
                box_mask_keys=["box_mask"],
                min_fg_label=0,
            ),
            Rotate90d(keys=image_key, k=3),
            Flipd(image_key, 1),
            EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
            EnsureTyped(keys=[label_key], dtype=int),
        ]),

        "dbt": Compose([
            LoadImaged(keys=[image_key], meta_key_postfix="meta_dict"),
            EnsureChannelFirstd(keys=[image_key]),
            EnsureTyped(keys=[label_key,box_key], dtype=torch.long, allow_missing_keys=True),
            EnsureTyped(keys=[image_key], dtype=torch.float16),
            BoxToMaskd(
                box_keys=[box_key],
                label_keys=[label_key],
                box_mask_keys=["box_mask"],
                box_ref_image_keys=image_key,
                min_fg_label=0,
                ellipse_mask=True,
                allow_missing_keys=True,
            ),
            Resized(
                keys=[image_key, "box_mask"],
                spatial_size=resize,
                mode=('bilinear','nearest'),
                allow_missing_keys=True
            ),
            MaskToBoxd(
                box_keys=[box_key],
                label_keys=[label_key],
                box_mask_keys=["box_mask"],
                min_fg_label=0,
                allow_missing_keys=True
            ),
            Rotate90d(keys=image_key, k=3),
            Flipd(image_key, 1),
            EnsureTyped(keys=[image_key, box_key], dtype=torch.float32, allow_missing_keys=True),
            EnsureTyped(keys=[label_key], dtype=int, allow_missing_keys=True),
        ])
    }
    return transforms[dataset_name]