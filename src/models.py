import torch
import torch.nn as nn
import torchvision.models as models
import timm


class FontClassifierModel(nn.Module):
    def __init__(self, num_classes, style_mapping=None, backbone_name='convnextv2_tiny.fcmae_ft_in1k', dropout=0.4):
        super(FontClassifierModel, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, features_only=True)
        last_ch = self.backbone.feature_info.channels()[-1]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.neck = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(last_ch, 512),
            nn.ReLU(inplace=True),
        )
        self.head_style = nn.Linear(512, num_classes)
        self.head_angle_vec = nn.Linear(512, 2)
        self.style_mapping = style_mapping

    def forward(self, x):
        feats = self.backbone(x)[-1]
        x = self.pool(feats)
        x = torch.flatten(x, 1)
        x = self.neck(x)
        style_logits = self.head_style(x)
        angle_vec = self.head_angle_vec(x)
        angle_vec = torch.nn.functional.normalize(angle_vec, dim=-1)

        # Convert angle vector back to degrees to match the expected output format
        s, c = angle_vec[..., 0], angle_vec[..., 1]
        angle_deg = torch.atan2(s, c) * 180.0 / 3.141592653589793

        return {"angle": angle_deg, "style": style_logits}


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
