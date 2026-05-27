import torch
import torch.nn as nn


class PatchTransformer1D(nn.Module):
    """ViT-style patch transformer adapted for 1D I/Q signals.

    Divides the signal into non-overlapping patches, linearly embeds each patch,
    prepends a learnable CLS token, adds learnable positional encodings, and
    encodes with a stack of pre-norm TransformerEncoder layers. Classification
    reads from the CLS token.

    Follows the ViT architecture directly, substituting 1D patch flattening for
    2D image patch extraction. Attention mechanism from the original Transformer.

    Citations:
        Dosovitskiy, Alexey, et al. "An Image Is Worth 16x16 Words: Transformers
        for Image Recognition at Scale." *International Conference on Learning
        Representations (ICLR)*, 2021.
        https://arxiv.org/abs/2010.11929

        Vaswani, Ashish, et al. "Attention Is All You Need." *Advances in Neural
        Information Processing Systems (NeurIPS)*, vol. 30, 2017.
        https://arxiv.org/abs/1706.03762
    """

    def __init__(
        self,
        n_classes: int,
        in_channels: int = 2,
        n_samples: int = 2048,
        patch_size: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        n_patches = n_samples // patch_size
        patch_dim = in_channels * patch_size
        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        P = self.patch_size
        n_patches = T // P
        x = x[:, :, : n_patches * P].reshape(B, C, n_patches, P)
        x = x.permute(0, 2, 1, 3).reshape(B, n_patches, C * P)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed[:, : n_patches + 1]
        x = self.transformer(x)
        return self.classifier(self.norm(x[:, 0]))
