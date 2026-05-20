"""CNN model definitions for the RQA classification project."""

import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Encoder1Ch(nn.Module):
    """ResNet-18 encoder adapted to one-channel grayscale inputs.

    Parameters
    ----------
    emb_dim : int, default 128
        Output embedding dimension produced by the projection head.
    pretrained : bool, default True
        Whether to initialize from ImageNet pretrained weights.
    dropout : float, default 0.3
        Dropout probability applied in the projection head.
    """
    def __init__(self, emb_dim=128, pretrained=True, dropout=0.3):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = models.resnet18(weights=weights)

        # --- change first conv from 3->64 to 1->64 (keep pretrained info) ---
        old_conv = m.conv1  # (64,3,7,7)
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)  # average RGB -> 1ch

        m.conv1 = new_conv

        # backbone up to global pool
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # output (N,512,1,1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """Project an image batch into the embedding space.

        Parameters
        ----------
        x : torch.Tensor
            Batch of grayscale images with shape ``(N, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Batch of embeddings with shape ``(N, emb_dim)``.
        """
        f = self.backbone(x)
        z = self.head(f)
        return z

class PlotLevelCNN(nn.Module):
    """Binary classifier built on top of the grayscale ResNet encoder.

    Parameters
    ----------
    emb_dim : int, default 128
        Embedding dimension produced by the encoder.
    num_classes : int, default 2
        Number of output classes.
    dropout : float, default 0.3
        Dropout probability passed to the encoder.
    """
    def __init__(self, emb_dim=128, num_classes=2, dropout=0.3):
        super().__init__()
        self.encoder = ResNet18Encoder1Ch(emb_dim=emb_dim, dropout=dropout)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        """Run a forward pass and return class logits.

        Parameters
        ----------
        x : torch.Tensor
            Batch of grayscale images with shape ``(B, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Class logits with shape ``(B, num_classes)``.
        """
        feats = self.encoder(x)
        logits = self.classifier(feats)
        return logits

    def freeze_backbone(self):
        """Freeze the encoder backbone parameters.

        Returns
        -------
        None
            Backbone parameters are marked as non-trainable in place.
        """
        for p in self.encoder.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_block_only(self):
        """Unfreeze the final residual block only.

        Returns
        -------
        None
            The last backbone block becomes trainable.
        """
        for p in self.encoder.backbone[7].parameters():
            p.requires_grad = True

    def unfreeze_last_stage(self):
        """Unfreeze the last layer inside the final residual block.

        Returns
        -------
        None
            The final stage parameters become trainable.
        """
        for p in self.encoder.backbone[7][-1].parameters():
            p.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreeze the full encoder backbone.

        Returns
        -------
        None
            All backbone parameters become trainable.
        """
        for p in self.encoder.backbone.parameters():
            p.requires_grad = True

