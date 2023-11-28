from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from multimodal_breast_analysis.models.backbones import resnet18, resnet50, resnet101


def faster_rcnn(parameters):
    backbones = {
        "resnet18" : resnet18,
        "resnet50" : resnet50,
        "resnet101" : resnet101
    } 
    backbone = backbones[parameters["backbone"]]()
    anchor_generator = AnchorGenerator(
        sizes=parameters["anchors_sizes"],
        aspect_ratios=parameters["anchors_aspect_ratios"]
    )
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=parameters["align_size"],
        sampling_ratio=parameters["align_sample_ratio"],
    )
    return FasterRCNN(
        backbone,
        num_classes=parameters["num_classes"],
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )