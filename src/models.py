import torch
import torch.nn as nn
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
        self.head_size = nn.Linear(512, 1)
        self.style_mapping = style_mapping

    def forward(self, x):
        feats = self.backbone(x)[-1]
        x = self.pool(feats)
        x = torch.flatten(x, 1)
        x = self.neck(x)

        style_logits = self.head_style(x)

        angle_vec = self.head_angle_vec(x)
        angle_vec = torch.nn.functional.normalize(angle_vec, dim=-1)
        s, c = angle_vec[..., 0], angle_vec[..., 1]
        angle_deg = torch.atan2(s, c) * 180.0 / 3.141592653589793

        size_pred = self.head_size(x).squeeze(-1)

        return {"angle": angle_deg, "style": style_logits, "size": size_pred}
