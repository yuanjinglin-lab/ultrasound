import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from models.encoder import Encoder  
from models.convlstm import ConvLSTM2d
from nets.mobilenetv2 import MobileNetV2
from models.rexnetv1 import ReXNetV1
from models.swin_transformer import SwinTransformer

__all__ = [
    "FusionModule",
    "UnifiedModel",
]


class FusionModule(nn.Module):

    def __init__(self, method: str = "none", input_dim: int = 512):
        super().__init__()
        self.method = method.lower()
        self.input_dim = input_dim

        if self.method == "tensor":
            self.tensor_fusion = nn.Sequential(
                nn.Linear(input_dim * input_dim, input_dim),
                nn.ReLU(),
            )
        elif self.method == "weighted":
            self.weights = nn.Parameter(torch.ones(2))
        elif self.method == "cross":
            self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8, batch_first=True)
        elif self.method in ("none", "identity"):
            pass  
        elif self.method == "concat":
            pass 
        else:
            raise ValueError(f"Unsupported fusion method: {method}")

    def forward(self, x1: torch.Tensor, x2: torch.Tensor | None = None):
        if self.method in ("none", "identity"):
            return x1  

        if x2 is None:
            x2 = x1.clone()

        if self.method == "concat":
            return torch.cat((x1, x2), dim=1)
        elif self.method == "tensor":
            B = x1.size(0)
            return self.tensor_fusion((x1 * x2).view(B, -1))
        elif self.method == "weighted":
            w = F.softmax(self.weights, dim=0)
            return w[0] * x1 + w[1] * x2
        elif self.method == "cross":
            x1 = x1.unsqueeze(1)  # [B, 1, C]
            x2 = x2.unsqueeze(1)
            out, _ = self.attn(x1, x2, x2)
            return out.squeeze(1)
        else:
            raise RuntimeError("Unexpected fusion method")



def build_encoder(name: str, c_in: int) -> nn.Module:
    name = name.lower()
    if name == "mobilenetv2":
        return MobileNetV2()
    elif name == "resnet18":
        enc = models.resnet18(norm_layer=nn.InstanceNorm2d, weights=None)
        enc.conv1 = nn.Conv2d(c_in, 64, kernel_size=3, stride=2, padding=3, bias=False)
        enc.fc = nn.Identity()
        return enc
    elif name == "rexnetv1":
        return ReXNetV1(input_ch=c_in)
    elif name == "swin":
        return SwinTransformer(hidden_dim=128, layers=(2, 2, 18, 2), heads=(3, 6, 12, 24), channels=c_in)
    else:
        raise ValueError(f"Unknown encoder: {name}")


class UnifiedModel(nn.Module):

    def __init__(
        self,
        c_in: int = 20,
        num_classes: int = 2,
        encoder_name: str = "mobilenetv2",
        fusion_method: str = "none",
        pooling: str = "max",
    ):
        super().__init__()
        self.encoder = build_encoder(encoder_name, c_in)

        self.hidden_dim = 512
        self.conv_lstm = ConvLSTM2d(
            input_dim=self.hidden_dim,
            hidden_dim=[self.hidden_dim] * 3,
            kernel_size=(3, 3),
            num_layers=3,
            batch_first=True,
            bias=True,
            return_all_layers=False,
        )

        pooling = pooling.lower()
        if pooling == "max":
            self.pool = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
            self.post_spatial = 4 
        elif pooling in ("avg", "adaptiveavg", "adaptive"):
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.post_spatial = 1 
        else:
            raise ValueError("pooling must be 'max' or 'avg'")

        self.fusion = FusionModule(method=fusion_method, input_dim=self.hidden_dim * self.post_spatial)
        fusion_out_dim = self.hidden_dim * self.post_spatial
        if fusion_method == "concat":
            fusion_out_dim *= 2

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_classes),
        ) if fusion_method != "concat" else nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_classes),
        )

    def forward(self, x: torch.Tensor):
        """x: [B, T, C, H, W]"""
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        feat = self.encoder(x)                
        feat = feat.view(B, T, *feat.shape[1:])  

        _, layer_out = self.conv_lstm(feat)  
        last_state = layer_out[-1][0]         
        pooled = self.pool(last_state)        
        pooled = pooled.flatten(1)            
        fused = self.fusion(pooled)
        logits = self.classifier(fused)
        return logits


if __name__ == "__main__":
    B, T, C, H, W = 2, 3, 20, 256, 256
    x = torch.randn(B, T, C, H, W)
    model = UnifiedModel(c_in=C, num_classes=2, encoder_name="mobilenetv2", fusion_method="concat", pooling="avg")
    out = model(x)
    print(out.shape)
