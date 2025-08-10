import torch
import torch.nn as nn
import torchvision.models as models
import timm


class FontClassifierModel(nn.Module):
    def __init__(self, num_classes, style_mapping=None):
        super(FontClassifierModel, self).__init__()
        self.backbone = timm.create_model(
            'convnextv2_tiny.fcmae_ft_in1k',
            pretrained=True,
            features_only=True
        )
        last_channel = 768  # ConvNeXT V2 Tiny의 출력 채널
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(last_channel, 512), nn.ReLU())
        self.angle_head = nn.Linear(512, 1)
        self.style_head = nn.Linear(512, num_classes)
        self.style_mapping = style_mapping

    def forward(self, x):
        features = self.backbone(x)
        x = features[-1]
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        angle_output = self.angle_head(x).squeeze(-1)
        style_output = self.style_head(x)
        return {"angle": angle_output, "style": style_output}


class FontSizeModel(nn.Module):
    def __init__(self):
        super(FontSizeModel, self).__init__()
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, 512), nn.ReLU())
        self.size_head = nn.Linear(512, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.size_head(x).squeeze(-1)
