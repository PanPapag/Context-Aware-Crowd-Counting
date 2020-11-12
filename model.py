import collections
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

class ContextualModule(nn.Module):
    def __init__(self, features, out_channels=512, sizes=[1, 2, 3, 6]):
        super(ContextualModule, self).__init__()
        self.scales = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(size, size)),
            nn.Conv2d(features, features, kernel_size=1, bias=False)) for size in sizes])
        self.bottleneck = nn.Conv2d(features * 2, out_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.weight_net = nn.Conv2d(features, features, kernel_size=1)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        multi_scales = [F.upsample(input=scale(x), size=(h, w), mode='bilinear') for scale in self.scales]
        weights = [F.sigmoid(self.weight_net(x - scale_feature)) for scale_feature in multi_scales]
        overall_features = [(multi_scales[0]*weights[0]+multi_scales[1]*weights[1]+multi_scales[2]*weights[2]+multi_scales[3]*weights[3])/(weights[0]+weights[1]+weights[2]+weights[3])]+[x]
        bottle = self.bottleneck(torch.cat(overall_features, 1))
        return self.relu(bottle)

class CANNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CANNet, self).__init__()
        self.contextual = ContextualModule(512, 512)
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
        )
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            fsd = collections.OrderedDict()
            for i in range(len(self.frontend.state_dict().items())):
                temp_key = list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)

    def forward(self,x):
        x = self.frontend(x)
        x = self.contextual(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# Test code
if __name__=="__main__":
    cannet = CANNet()
    print(cannet)
    input = torch.ones((1,3,256,256))
    out = cannet(input)
    print(out.shape, out.mean())
