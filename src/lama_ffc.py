"""만화 특화 LaMa 인페인팅 (FFC ResNet Generator).

dreMaz/AnimeMangaInpainting의 lama_large_512px.ckpt(만화/애니메이션 데이터로
파인튜닝된 big-lama)를 로드하기 위한 최소 구현.

아키텍처는 원본 LaMa(advimman/lama, Apache-2.0)의 FFCResNetGenerator를
체크포인트 state_dict 키와 1:1로 일치하도록 재작성했다:
  model.0  ReflectionPad2d(3)
  model.1  FFC_BN_ACT(4→64, k7)           # ratio 0/0
  model.2  FFC_BN_ACT(64→128, k3 s2)      # ratio 0/0
  model.3  FFC_BN_ACT(128→256, k3 s2)     # ratio 0/0
  model.4  FFC_BN_ACT(256→512, k3 s2)     # ratio 0/0.75
  model.5..22  FFCResnetBlock(512) ×18    # ratio 0.75/0.75
  model.23 ConcatTupleLayer
  model.24..32 ConvTranspose2d + BN + ReLU ×3 (512→256→128→64)
  model.33 ReflectionPad2d(3), model.34 Conv2d(64→3, k7), model.35 Sigmoid
"""
import logging
import os

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

HF_REPO_ID = "dreMaz/AnimeMangaInpainting"
HF_FILENAME = "lama_large_512px.ckpt"


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        self.groups = groups
        self.conv_layer = nn.Conv2d(in_channels * 2, out_channels * 2, kernel_size=1,
                                    stride=1, padding=0, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch = x.shape[0]
        # FFT는 half에서 불안정하므로 이 블록은 fp32 고정으로 돈다.
        ffted = torch.fft.rfftn(x.float(), dim=(-2, -1), norm="ortho")
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)          # (b, c, h, w/2+1, 2)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()              # (b, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])            # (b, 2c, h, w/2+1)

        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        output = torch.fft.irfftn(ffted, s=x.shape[-2:], dim=(-2, -1), norm="ortho")
        return output.to(x.dtype)


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.stride = stride
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2) if stride == 2 else nn.Identity()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)
        output = self.conv2(x + output)
        return output


class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, padding_type="reflect"):
        super().__init__()
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size, stride, padding,
                              dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size, stride, padding,
                              dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size, stride, padding,
                              dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2)

    def forward(self, x):
        x_l, x_g = x if isinstance(x, tuple) else (x, 0)
        out_xl, out_xg = 0, 0
        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)
        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0, padding_type="reflect"):
        super().__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size, ratio_gin, ratio_gout,
                       stride=stride, padding=padding, padding_type=padding_type)
        out_cg = int(out_channels * ratio_gout)
        lnorm = nn.Identity if ratio_gout == 1 else nn.BatchNorm2d
        gnorm = nn.Identity if ratio_gout == 0 else nn.BatchNorm2d
        self.bn_l = lnorm(out_channels - out_cg)
        self.bn_g = gnorm(out_cg)
        lact = nn.Identity if ratio_gout == 1 else nn.ReLU
        gact = nn.Identity if ratio_gout == 0 else nn.ReLU
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(self, dim, ratio_gin=0.75, ratio_gout=0.75, padding_type="reflect"):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, ratio_gin=ratio_gin,
                                ratio_gout=ratio_gout, padding=1, padding_type=padding_type)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, ratio_gin=ratio_gin,
                                ratio_gout=ratio_gout, padding=1, padding_type=padding_type)

    def forward(self, x):
        x_l, x_g = x if isinstance(x, tuple) else (x, 0)
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        return id_l + x_l, id_g + x_g


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        x_l, x_g = x
        return torch.cat((x_l, x_g), dim=1)


def _build_generator(input_nc=4, output_nc=3, ngf=64, n_downsampling=3, n_blocks=18):
    layers = [
        nn.ReflectionPad2d(3),
        FFC_BN_ACT(input_nc, ngf, kernel_size=7, ratio_gin=0, ratio_gout=0, padding=0),
    ]
    for i in range(n_downsampling):
        mult = 2 ** i
        ratio_gout = 0.75 if i == n_downsampling - 1 else 0
        layers.append(FFC_BN_ACT(ngf * mult, ngf * mult * 2, kernel_size=3,
                                 ratio_gin=0, ratio_gout=ratio_gout, stride=2, padding=1))

    feat_dim = ngf * (2 ** n_downsampling)
    for _ in range(n_blocks):
        layers.append(FFCResnetBlock(feat_dim))

    layers.append(ConcatTupleLayer())
    for i in range(n_downsampling):
        mult = 2 ** (n_downsampling - i)
        layers += [
            nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ngf * mult // 2),
            nn.ReLU(True),
        ]
    layers += [
        nn.ReflectionPad2d(3),
        nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
        nn.Sigmoid(),
    ]

    class _Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    return _Generator()


class MangaLama:
    """SimpleLama와 동일한 호출 규약을 제공하는 만화 특화 LaMa 래퍼.

    `.model(img_batch, mask_batch)` — 0..1 float (B,3,H,W) 이미지와 (B,1,H,W)
    이진 마스크를 받아 합성 완료된 (B,3,H,W) 결과를 반환한다. H/W는 8의 배수.
    """

    def __init__(self, generator: nn.Module, device: str):
        self.generator = generator
        self.device = torch.device(device)

    def model(self, img_batch: torch.Tensor, mask_batch: torch.Tensor) -> torch.Tensor:
        mask_batch = (mask_batch > 0.5).to(img_batch.dtype)
        masked_img = img_batch * (1.0 - mask_batch)
        net_input = torch.cat([masked_img, mask_batch], dim=1)
        prediction = self.generator(net_input)
        return mask_batch * prediction + (1.0 - mask_batch) * img_batch


def _resolve_checkpoint(model_path: str) -> str:
    if model_path and os.path.exists(model_path):
        return model_path
    logger.info("만화 인페인팅 체크포인트가 없어 Hugging Face에서 내려받습니다: %s", HF_REPO_ID)
    from huggingface_hub import hf_hub_download
    local_dir = os.path.dirname(model_path) or "data/models/anime_lama"
    return hf_hub_download(HF_REPO_ID, HF_FILENAME, local_dir=local_dir)


def load_manga_lama(model_path: str, device: str) -> MangaLama:
    resolved_path = _resolve_checkpoint(model_path)
    checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("gen_state_dict", checkpoint)

    generator = _build_generator()
    generator.load_state_dict(state_dict, strict=True)
    generator.to(device)
    generator.eval()
    logger.info("Manga-specialized LaMa generator loaded from %s", resolved_path)
    return MangaLama(generator, device)
