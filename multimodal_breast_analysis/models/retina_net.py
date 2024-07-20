from torchvision.models.detection import RetinaNet
from torchvision.models.detection.anchor_utils import AnchorGenerator
from multimodal_breast_analysis.models.backbones import resnet18, resnet34, resnet50, resnet101, swin_transformer


def retina_net(parameters):
    """
    Creates a retinanet model with the defined parameters
    Args:
        parameters: dict including model paramters
    """
    backbones = {
        "resnet18" : resnet18,
        "resnet34" : resnet34,
        "resnet50" : resnet50,
        "resnet101" : resnet101,
        "swin_transformer" : swin_transformer
    } 
    backbone = backbones[parameters["backbone"]]()
    anchor_generator = AnchorGenerator(
        sizes=parameters["anchors_sizes"],
        aspect_ratios=parameters["anchors_aspect_ratios"]*len(parameters["anchors_sizes"]) 
    )
    return RetinaNet(
        backbone,
        num_classes=parameters["num_classes"],
        anchor_generator=anchor_generator,
    )