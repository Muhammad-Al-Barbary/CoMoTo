from torchvision.models.resnet import ResNet, BasicBlock

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
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x
    
def resnet18():
    resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
    resnet18.out_channels = 512
    return resnet18
def resnet50():
    resnet50 = ResNet(BasicBlock, [3, 4, 6, 3])
    resnet50.out_channels = 512
    return resnet50
def resnet101():
    resnet101 = ResNet(BasicBlock, [3, 4, 23, 3])
    resnet101.out_channels = 512
    return resnet101