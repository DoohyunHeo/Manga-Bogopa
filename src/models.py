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

    def __init__(self, backbone_name: str, pretrained: bool = False):
        super().__init__()
        self.backbone_name = backbone_name
        self.uses_features_only = False

        if self._prefer_token_adapter(backbone_name):
            self.model = self._create_token_backbone(backbone_name, pretrained)
            self.out_channels = int(getattr(self.model, 'num_features'))
        else:
            try:
                self.model = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
                self.out_channels = int(self.model.feature_info.channels()[-1])
                self.uses_features_only = True
            except RuntimeError as exc:
                if pretrained and self._is_missing_pretrained_weights_error(exc):
                    self.model = timm.create_model(backbone_name, pretrained=False, features_only=True)
                    self.out_channels = int(self.model.feature_info.channels()[-1])
                    self.uses_features_only = True
                else:
                    raise
            except Exception:
                self.model = self._create_token_backbone(backbone_name, pretrained)
                self.out_channels = int(getattr(self.model, 'num_features'))

    @staticmethod
    def _prefer_token_adapter(backbone_name: str) -> bool:
        lowered = backbone_name.lower()
        token_markers = ('dinov2', 'eva02', 'siglip', 'beit', 'deit', 'vit_', 'naflexvit')
        return any(marker in lowered for marker in token_markers)

    @staticmethod
    def _is_missing_pretrained_weights_error(exc: Exception) -> bool:
        return "no pretrained weights exist for" in str(exc).lower()

    @staticmethod
    def _create_token_backbone(backbone_name: str, pretrained: bool):
        try:
            return timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool='',
                dynamic_img_size=True,
            )
        except RuntimeError as exc:
            if pretrained and TimmSpatialBackbone._is_missing_pretrained_weights_error(exc):
                pretrained = False
            else:
                raise
        try:
            return timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool='',
                dynamic_img_size=True,
            )
        except TypeError:
            return timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool='',
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
        size_target_transform='identity',
    ):
        super(FontClassifierModel, self).__init__()
        self.backbone = TimmSpatialBackbone(backbone_name, pretrained=False)
        last_ch = self.backbone.out_channels
        self.pool_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_max = nn.AdaptiveMaxPool2d((1, 1))
        self.neck = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(last_ch, 512),
            nn.ReLU(inplace=True),
        )
        self.head_style = nn.Linear(512, num_classes)
        self.head_expressive = nn.Linear(512, 1)
        self.head_angle_vec = nn.Linear(512, 2)
        self.size_context = nn.Linear(last_ch * 2, 512)
        nn.init.zeros_(self.size_context.weight)
        nn.init.zeros_(self.size_context.bias)
        self.head_size = nn.Linear(512, 1)
        self.style_mapping = style_mapping
        normalized_transform = str(size_target_transform or 'identity').lower()
        self.size_target_transform = normalized_transform if normalized_transform in {'identity', 'log'} else 'identity'

    def forward(self, x):
        feats = self.backbone(x)
        pooled_avg = torch.flatten(self.pool_avg(feats), 1)
        pooled_max = torch.flatten(self.pool_max(feats), 1)
        x = self.neck(pooled_avg)
        size_features = x + self.size_context(torch.cat([pooled_avg, pooled_max], dim=1))

        style_logits = self.head_style(x)
        expressive_logits = self.head_expressive(x).squeeze(-1)

        angle_vec = self.head_angle_vec(x)
        angle_vec = torch.nn.functional.normalize(angle_vec, dim=-1)
        s, c = angle_vec[..., 0], angle_vec[..., 1]
        angle_deg = torch.atan2(s, c) * 180.0 / 3.141592653589793

        size_raw = self.head_size(size_features).squeeze(-1)
        if self.size_target_transform == 'log':
            size_pred = torch.exp(size_raw.clamp(max=6.0))
        else:
            size_pred = size_raw

        return {
            "angle": angle_deg,
            "style": style_logits,
            "expressive": expressive_logits,
            "size_raw": size_raw,
            "size": size_pred,
        }
