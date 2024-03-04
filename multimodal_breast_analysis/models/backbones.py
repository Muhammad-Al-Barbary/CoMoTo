from torchvision.models.resnet import ResNet, BasicBlock, ResNet18_Weights, ResNet50_Weights, ResNet101_Weights, ResNet34_Weights, Bottleneck
from monai.networks.nets.swin_unetr import SwinTransformer
from monai.utils import ensure_tuple_rep

class ResNet(ResNet):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    

class SwinTransformer(SwinTransformer):
    def forward(self, x, normalize=True):
            x = self.patch_embed(x)
            x = self.pos_drop(x)
            x = self.layers1[0](x.contiguous())
            x = self.layers2[0](x.contiguous())
            x = self.layers3[0](x.contiguous())
            x = self.layers4[0](x.contiguous())
            x = self.proj_out(x, normalize)
            return x
    

def resnet18():
    resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
    resnet18.load_state_dict(ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True))
    resnet18.out_channels = 512
    return resnet18


def resnet34():
    resnet34 = ResNet(BasicBlock, [3, 4, 6, 3])
    resnet34.load_state_dict(ResNet34_Weights.IMAGENET1K_V1.get_state_dict(progress=True))
    resnet34.out_channels = 512
    return resnet34


def resnet50():
    resnet50 = ResNet(Bottleneck, [3, 4, 6, 3])
    resnet50.load_state_dict(ResNet50_Weights.IMAGENET1K_V1.get_state_dict(progress=True))
    resnet50.out_channels = 2048
    return resnet50


def resnet101():
    resnet101 = ResNet(Bottleneck, [3, 4, 23, 3])
    resnet101.load_state_dict(ResNet101_Weights.IMAGENET1K_V1.get_state_dict(progress=True))
    resnet101.out_channels = 2048
    return resnet101


def swin_transformer():
    spatial_dims = 2
    in_chans = 3
    embed_dim = 36
    num_heads = (6, 12, 24, 48)
    depths = (2, 2, 2, 2)
    swin_transformer = SwinTransformer(
                        spatial_dims=spatial_dims, in_chans=in_chans, embed_dim=embed_dim, window_size=ensure_tuple_rep(7, spatial_dims),
                        patch_size=ensure_tuple_rep(2, spatial_dims), depths=depths, num_heads=num_heads
                        )
    swin_transformer.out_channels = 16*embed_dim
    return swin_transformer