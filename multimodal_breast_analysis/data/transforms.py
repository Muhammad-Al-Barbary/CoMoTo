from monai.transforms import (
    Compose,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Rotate90d,
    Flipd,
    Resized,
    CropForegroundd,
    NormalizeIntensityd,
    ThresholdIntensityd,
    RandFlipd,
    Rand2DElasticd
)
from monai.apps.detection.transforms.dictionary import (
    AffineBoxToImageCoordinated,
    AffineBoxToWorldCoordinated,
    BoxToMaskd,
    ClipBoxToImaged,
    ConvertBoxToStandardModed,
    MaskToBoxd,
    RandCropBoxByPosNegLabeld,
    RandFlipBoxd,
    RandRotateBox90d,
    RandZoomBoxd,
    ConvertBoxModed,
    # StandardizeEmptyBoxd,
)
import torch

from multimodal_breast_analysis.configs.configs import config


image_key = "image"
box_key = "boxes"
label_key = "labels"
resize = config.transforms['size']

def train_transforms(dataset_name):
    transforms = {
                    "penn_fudan": Compose([
                                    LoadImaged(keys=[image_key], meta_key_postfix="meta_dict"),
                                    EnsureChannelFirstd(keys=[image_key]),
                                    EnsureTyped(keys=[image_key], dtype=torch.float32),
                                    EnsureTyped(keys=[label_key,box_key], dtype=torch.long),
                                    # StandardizeEmptyBoxd(box_keys=[box_key], box_ref_image_keys=image_key),
                                    # Orientationd(keys=[image_key], axcodes="RAS"),
                                    # intensity_transform,
                                    EnsureTyped(keys=[image_key], dtype=torch.float16),
                                    # ConvertBoxToStandardModed(box_keys=[box_key], mode="xxyy"),
                                    # AffineBoxToImageCoordinated(
                                    #     box_keys=[box_key],
                                    #     box_ref_image_keys=image_key,
                                    #     image_meta_key_postfix="meta_dict",
                                    #     affine_lps_to_ras=affine_lps_to_ras,
                                    # ),
                                    # RandCropBoxByPosNegLabeld(
                                    #     image_keys=[image_key],
                                    #     box_keys=box_key,
                                    #     label_keys=label_key,
                                    #     spatial_size=patch_size,
                                    #     whole_box=True,
                                    #     num_samples=batch_size,
                                    #     pos=1,
                                    #     neg=1,
                                    # ),
                                    # RandZoomBoxd(
                                    #     image_keys=[image_key],
                                    #     box_keys=[box_key],
                                    #     box_ref_image_keys=[image_key],
                                    #     prob=0.2,
                                    #     min_zoom=0.7,
                                    #     max_zoom=1.4,
                                    #     padding_mode="constant",
                                    #     keep_size=True,
                                    # ),
                                    # ClipBoxToImaged(
                                    #     box_keys=box_key,
                                    #     label_keys=[label_key],
                                    #     box_ref_image_keys=image_key,
                                    #     remove_empty=True,
                                    # ),
                                    # RandFlipBoxd(
                                    #     image_keys=[image_key],
                                    #     box_keys=[box_key],
                                    #     box_ref_image_keys=[image_key],
                                    #     prob=0.5,
                                    #     spatial_axis=0,
                                    # ),
                                    # RandFlipBoxd(
                                    #     image_keys=[image_key],
                                    #     box_keys=[box_key],
                                    #     box_ref_image_keys=[image_key],
                                    #     prob=0.5,
                                    #     spatial_axis=1,
                                    # ),
                                    # RandFlipBoxd(
                                    #     image_keys=[image_key],
                                    #     box_keys=[box_key],
                                    #     box_ref_image_keys=[image_key],
                                    #     prob=0.5,
                                    #     spatial_axis=2,
                                    # ),
                                    # RandRotateBox90d(
                                    #     image_keys=[image_key],
                                    #     box_keys=[box_key],
                                    #     box_ref_image_keys=[image_key],
                                    #     prob=0.75,
                                    #     max_k=3,
                                    #     spatial_axes=(0, 1),
                                    # ),
                                    BoxToMaskd(
                                        box_keys=[box_key],
                                        label_keys=[label_key],
                                        box_mask_keys=["box_mask"],
                                        box_ref_image_keys=image_key,
                                        min_fg_label=0,
                                        ellipse_mask=False,
                                    ),
                                    Resized(
                                        keys=[image_key, "box_mask"],
                                        spatial_size=resize,
                                        mode=('bilinear','nearest')
                                    ),
                                    # RandRotated(
                                    #     keys=[image_key, "box_mask"],
                                    #     mode=["nearest", "nearest"],
                                    #     prob=0.2,
                                    #     range_x=np.pi / 6,
                                    #     range_y=np.pi / 6,
                                    #     range_z=np.pi / 6,
                                    #     keep_size=True,
                                    #     padding_mode="zeros",
                                    # ),
                                    MaskToBoxd(
                                        box_keys=[box_key],
                                        label_keys=[label_key],
                                        box_mask_keys=["box_mask"],
                                        min_fg_label=0,
                                    ),
                                    # DeleteItemsd(keys=["box_mask"]),
                                    # RandGaussianNoised(keys=[image_key], prob=0.1, mean=0, std=0.1),
                                    # RandGaussianSmoothd(
                                    #     keys=[image_key],
                                    #     prob=0.1,
                                    #     sigma_x=(0.5, 1.0),
                                    #     sigma_y=(0.5, 1.0),
                                    #     sigma_z=(0.5, 1.0),
                                    # ),
                                    # RandScaleIntensityd(keys=[image_key], prob=0.15, factors=0.25),
                                    # RandShiftIntensityd(keys=[image_key], prob=0.15, offsets=0.1),
                                    # RandAdjustContrastd(keys=[image_key], prob=0.3, gamma=(0.7, 1.5)),
                                    Rotate90d(keys=image_key, k=3),
                                    Flipd(image_key, 1),
                                    EnsureTyped(keys=[image_key, box_key], dtype=torch.float32),
                                    EnsureTyped(keys=[label_key], dtype=int),
                                    ]),


                    "omidb": Compose([
                                    LoadImaged(keys=[image_key], meta_key_postfix="meta_dict"),
                                    EnsureChannelFirstd(keys=[image_key]),
                                    EnsureTyped(keys=[label_key,box_key], dtype=torch.long),
                                    EnsureTyped(keys=[image_key], dtype=torch.float16),
                                    ThresholdIntensityd([image_key], threshold = 4094, above=False),
                                    # NormalizeIntensityd([image_key]),
                                    ClipBoxToImaged(
                                        box_keys=box_key,
                                        label_keys=[label_key],
                                        box_ref_image_keys=image_key,
                                        remove_empty=True,
                                    ),
                                    BoxToMaskd(
                                        box_keys=[box_key],
                                        label_keys=[label_key],
                                        box_mask_keys=["box_mask"],
                                        box_ref_image_keys=image_key,
                                        min_fg_label=0,
                                        ellipse_mask=True,
                                    ),
                                    CropForegroundd(
                                        keys=[image_key, "box_mask"],
                                        source_key=image_key
                                    ),
                                    Resized(
                                        keys=[image_key, "box_mask"],
                                        spatial_size=resize,
                                        mode=('bilinear','nearest')
                                    ),
                                    ###################### AUGMENTATIONS ######################
                                    RandFlipd(keys=[image_key, "box_mask"], prob = 0.5, spatial_axis = 0),
                                    RandFlipd(keys=[image_key, "box_mask"], prob = 0.5, spatial_axis = 1),
                                    RandRotated(keys=[image_key, "box_mask"], prob = 0.5, range_x = 0.25, mode = ['bilinear', 'nearest']),
                                    RandAdjustContrastd(keys=image_key, prob = 0.5, gamma = 2),
                                    RandGaussianNoised(keys=image_key, prob = 0.5),
                                    RandGaussianSmoothd(keys=image_key, prob = 0.5),
                                    ############################################################
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


                    "omidb_downsampled": Compose([
                                    LoadImaged(keys=[image_key], meta_key_postfix="meta_dict"),
                                    EnsureChannelFirstd(keys=[image_key]),
                                    EnsureTyped(keys=[label_key,box_key], dtype=torch.long),
                                    EnsureTyped(keys=[image_key], dtype=torch.float16),
                                    # NormalizeIntensityd([image_key]), #built-in in faster-rcnn
                                    # ClipBoxToImaged(
                                    #     box_keys=box_key,
                                    #     label_keys=[label_key],
                                    #     box_ref_image_keys=image_key,
                                    #     remove_empty=True,
                                    # ),
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
                                    ###################### AUGMENTATIONS ######################
                                    RandFlipd(keys=[image_key, "box_mask"], prob = 0.5, spatial_axis = 0),
                                    RandFlipd(keys=[image_key, "box_mask"], prob = 0.5, spatial_axis = 1),
                                    # RandRotated(keys=[image_key, "box_mask"], prob = 0.5, range_x = 0.35, mode = ['bilinear', 'nearest']),
                                    # RandAdjustContrastd(keys=image_key, prob = 0.5, gamma = 1.5),
                                    # RandGaussianNoised(keys=image_key, prob = 0.5),
                                    # RandGaussianSmoothd(keys=image_key, prob = 0.5),
                                    ############################################################
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
                    

                    "dbt_2d": Compose([
                                    LoadImaged(keys=[image_key], meta_key_postfix="meta_dict"),
                                    EnsureChannelFirstd(keys=[image_key]),
                                    EnsureTyped(keys=[label_key,box_key], dtype=torch.long),
                                    EnsureTyped(keys=[image_key], dtype=torch.float16),
                                    # NormalizeIntensityd([image_key]),
                                    # ClipBoxToImaged(
                                    #     box_keys=box_key,
                                    #     label_keys=[label_key],
                                    #     box_ref_image_keys=image_key,
                                    #     remove_empty=True,
                                    # ),
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
                                    ###################### AUGMENTATIONS ######################
                                    RandFlipd(keys=[image_key, "box_mask"], prob = 0.5, spatial_axis = 0),
                                    RandFlipd(keys=[image_key, "box_mask"], prob = 0.5, spatial_axis = 1),
                                    # RandRotated(keys=[image_key, "box_mask"], prob = 0.5, range_x = 0.35, mode = ['bilinear', 'nearest']),
                                    # RandAdjustContrastd(keys=image_key, prob = 0.75, gamma = 1.5),
                                    # RandGaussianNoised(keys=image_key, prob = 0.75),
                                    # RandGaussianSmoothd(keys=image_key, prob = 0.75),
                                    # Rand2DElasticd(keys = [image_key, "box_mask"], prob=0.75, spacing = (20,20), magnitude_range=(1,2), padding_mode='zeros', mode=['bilinear','nearest']),
                                    ############################################################
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


def test_transforms(dataset_name):
    transforms = {
                    "penn_fudan": Compose([
                                    LoadImaged(keys=[image_key], meta_key_postfix="meta_dict"),
                                    EnsureChannelFirstd(keys=[image_key]),
                                    EnsureTyped(keys=[image_key], dtype=torch.float32),
                                    EnsureTyped(keys=[label_key,box_key], dtype=torch.long),
                                    EnsureTyped(keys=[image_key], dtype=torch.float16),
                                    BoxToMaskd(
                                        box_keys=[box_key],
                                        label_keys=[label_key],
                                        box_mask_keys=["box_mask"],
                                        box_ref_image_keys=image_key,
                                        min_fg_label=0,
                                        ellipse_mask=False,
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
                                    

                    "omidb": Compose([
                                    LoadImaged(keys=[image_key], meta_key_postfix="meta_dict"),
                                    EnsureChannelFirstd(keys=[image_key]),
                                    EnsureTyped(keys=[label_key,box_key], dtype=torch.long),
                                    EnsureTyped(keys=[image_key], dtype=torch.float16),
                                    ClipBoxToImaged(
                                        box_keys=box_key,
                                        label_keys=[label_key],
                                        box_ref_image_keys=image_key,
                                        remove_empty=True,
                                    ),
                                    ThresholdIntensityd([image_key], threshold = 4094, above=False),
                                    # NormalizeIntensityd([image_key]),
                                    BoxToMaskd(
                                        box_keys=[box_key],
                                        label_keys=[label_key],
                                        box_mask_keys=["box_mask"],
                                        box_ref_image_keys=image_key,
                                        min_fg_label=0,
                                        ellipse_mask=True,
                                    ),
                                    CropForegroundd(
                                        keys=[image_key, "box_mask"],
                                        source_key=image_key
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


                    "omidb_downsampled": Compose([
                                    LoadImaged(keys=[image_key], meta_key_postfix="meta_dict"),
                                    EnsureChannelFirstd(keys=[image_key]),
                                    EnsureTyped(keys=[label_key,box_key], dtype=torch.long),
                                    EnsureTyped(keys=[image_key], dtype=torch.float16),
                                    # NormalizeIntensityd([image_key]),
                                    # ClipBoxToImaged(
                                    #     box_keys=box_key,
                                    #     label_keys=[label_key],
                                    #     box_ref_image_keys=image_key,
                                    #     remove_empty=True,
                                    # ),
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


                    "dbt_2d": Compose([
                                    LoadImaged(keys=[image_key], meta_key_postfix="meta_dict"),
                                    EnsureChannelFirstd(keys=[image_key]),
                                    EnsureTyped(keys=[label_key,box_key], dtype=torch.long, allow_missing_keys=True),
                                    EnsureTyped(keys=[image_key], dtype=torch.float16),
                                    # NormalizeIntensityd([image_key]),
                                    # ClipBoxToImaged(
                                    #     box_keys=box_key,
                                    #     label_keys=[label_key],
                                    #     box_ref_image_keys=image_key,
                                    #     remove_empty=True,
                                    # ),

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