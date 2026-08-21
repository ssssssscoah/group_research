import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFusionMLP(nn.Module):
    """Baseline: concatenate normalized image/text CLIP embeddings."""

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor):
        image_emb = F.normalize(image_emb, p=2, dim=-1)
        text_emb = F.normalize(text_emb, p=2, dim=-1)
        features = torch.cat([image_emb, text_emb], dim=-1)
        return self.classifier(features).squeeze(-1)


class ConsistencyAwareMLP(nn.Module):
    """
    Improved model using explicit cross-modal relation features.

    Base features:
        image embedding (512)
        text embedding  (512)

    Optional consistency features:
        cosine similarity       (1)
        absolute difference     (512)
        element-wise product    (512)
    """

    def __init__(
        self,
        use_cosine: bool = True,
        use_abs_diff: bool = True,
        use_interaction: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.use_cosine = use_cosine
        self.use_abs_diff = use_abs_diff
        self.use_interaction = use_interaction

        input_dim = 1024
        if use_cosine:
            input_dim += 1
        if use_abs_diff:
            input_dim += 512
        if use_interaction:
            input_dim += 512

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor):
        image_emb = F.normalize(image_emb, p=2, dim=-1)
        text_emb = F.normalize(text_emb, p=2, dim=-1)

        features = [image_emb, text_emb]

        if self.use_cosine:
            cosine = (image_emb * text_emb).sum(dim=-1, keepdim=True)
            features.append(cosine)

        if self.use_abs_diff:
            features.append(torch.abs(image_emb - text_emb))

        if self.use_interaction:
            features.append(image_emb * text_emb)

        fused = torch.cat(features, dim=-1)
        return self.classifier(fused).squeeze(-1)
