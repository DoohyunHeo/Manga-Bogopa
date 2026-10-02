from typing import Tuple

import torch
import torch.nn as nn
import timm


class TimmSpatialBackbone(nn.Module):
    """Unified backbone wrapper that handles both CNN (features_only) and
    ViT/token-based models (forward_features + spatial reshape).

    This must match the training-time TimmSpatialBackbone exactly so that
    checkpoint state_dict keys align (backbone.model.*).
    """

    def __init__(self, backbone_name: str):
        super().__init__()
        self.backbone_name = backbone_name
        self.uses_features_only = False

        if self._prefer_token_adapter(backbone_name):
            self.model = self._create_token_backbone(backbone_name)
            self.out_channels = int(getattr(self.model, 'num_features'))
        else:
            try:
                self.model = timm.create_model(backbone_name, pretrained=False, features_only=True)
                self.out_channels = int(self.model.feature_info.channels()[-1])
                self.uses_features_only = True
            except RuntimeError:
                raise  # a RuntimeError propagates; any other failure falls back to the token backbone
            except Exception:
                self.model = self._create_token_backbone(backbone_name)
                self.out_channels = int(getattr(self.model, 'num_features'))

    @staticmethod
    def _prefer_token_adapter(backbone_name: str) -> bool:
        lowered = backbone_name.lower()
        token_markers = ('dinov2', 'eva02', 'siglip', 'beit', 'deit', 'vit_', 'naflexvit')
        return any(marker in lowered for marker in token_markers)

    @staticmethod
    def _create_token_backbone(backbone_name: str):
        return timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            global_pool='',
            dynamic_img_size=True,
        )

    def _infer_token_grid(self, token_count: int, input_hw: Tuple[int, int]) -> Tuple[int, int] | None:
        patch_embed = getattr(self.model, 'patch_embed', None)
        if patch_embed is not None:
            grid_size = getattr(patch_embed, 'grid_size', None)
            if isinstance(grid_size, tuple) and len(grid_size) == 2 and grid_size[0] * grid_size[1] == token_count:
                return int(grid_size[0]), int(grid_size[1])

            patch_size = getattr(patch_embed, 'patch_size', None)
            if isinstance(patch_size, tuple) and len(patch_size) == 2:
                grid_h = max(1, input_hw[0] // patch_size[0])
                grid_w = max(1, input_hw[1] // patch_size[1])
                if grid_h * grid_w == token_count:
                    return grid_h, grid_w
            elif isinstance(patch_size, int) and patch_size > 0:
                grid_h = max(1, input_hw[0] // patch_size)
                grid_w = max(1, input_hw[1] // patch_size)
                if grid_h * grid_w == token_count:
                    return grid_h, grid_w

        side = int(round(token_count ** 0.5))
        if side * side == token_count:
            return side, side
        return None

    def _to_spatial_map(self, feats, input_hw: Tuple[int, int]) -> torch.Tensor:
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]

        if feats.dim() == 4:
            if feats.shape[1] == self.out_channels:
                return feats
            if feats.shape[-1] == self.out_channels:
                return feats.permute(0, 3, 1, 2).contiguous()

        if feats.dim() == 3:
            num_prefix_tokens = int(getattr(self.model, 'num_prefix_tokens', 0) or 0)
            if num_prefix_tokens > 0 and feats.shape[1] > num_prefix_tokens:
                feats = feats[:, num_prefix_tokens:, :]

            batch_size, token_count, channels = feats.shape
            if channels != self.out_channels:
                raise ValueError(
                    f"Unexpected token feature shape for {self.backbone_name}: {tuple(feats.shape)} "
                    f"(expected channel dim {self.out_channels})"
                )

            grid = self._infer_token_grid(token_count, input_hw)
            if grid is None:
                pooled = feats.mean(dim=1)
                return pooled.unsqueeze(-1).unsqueeze(-1)

            grid_h, grid_w = grid
            return feats.transpose(1, 2).reshape(batch_size, channels, grid_h, grid_w).contiguous()

        if feats.dim() == 2:
            return feats.unsqueeze(-1).unsqueeze(-1)

        raise ValueError(f"Unsupported feature shape from {self.backbone_name}: {tuple(feats.shape)}")

    def forward(self, x):
        if self.uses_features_only:
            feats = self.model(x)
        else:
            feats = self.model.forward_features(x)
        return self._to_spatial_map(feats, x.shape[-2:])


class FontClassifierModel(nn.Module):
    def __init__(
        self,
        num_classes,
        style_mapping=None,
        backbone_name='convnextv2_tiny.fcmae_ft_in1k',
        dropout=0.4,
    ):
        super(FontClassifierModel, self).__init__()
        self.backbone = TimmSpatialBackbone(backbone_name)
        last_ch = self.backbone.out_channels
        self.pool_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.neck = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(last_ch, 512),
            nn.ReLU(inplace=True),
        )
        self.head_style = nn.Linear(512, num_classes)
        self.style_mapping = style_mapping

    def forward(self, x):
        feats = self.backbone(x)
        pooled_avg = torch.flatten(self.pool_avg(feats), 1)
        x = self.neck(pooled_avg)
        return {"style": self.head_style(x)}
