from collections.abc import Mapping

import torch
import torch.nn as nn


class ModelWithPreprocessor(nn.Module):
    def __init__(self, backbone: nn.Module, preprocessor: nn.Module) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.preprocessor(x))


class MultiTaskModel(nn.Module):
    """Primary classifier plus auxiliary metadata heads from the pre-logit feature."""

    def __init__(
        self,
        backbone: nn.Module,
        auxiliary_classes: Mapping[str, int],
        hidden_dim: int = 0,
        uncertainty_weighting: bool = False,
    ) -> None:
        super().__init__()
        if not auxiliary_classes:
            raise ValueError("MultiTaskModel requires at least one auxiliary task.")
        self.backbone = backbone
        self._feature_layer_name, feature_layer = _last_linear(backbone)
        self._feature_dim = feature_layer.in_features
        self._features: torch.Tensor | None = None
        feature_layer.register_forward_pre_hook(self._capture_features)
        self.aux_heads = nn.ModuleDict(
            {
                name: _make_head(self._feature_dim, n_classes, hidden_dim)
                for name, n_classes in auxiliary_classes.items()
            }
        )
        self.loss_log_vars = nn.ParameterDict()
        if uncertainty_weighting:
            self.loss_log_vars["modulation"] = nn.Parameter(torch.zeros(()))
            for name in auxiliary_classes:
                self.loss_log_vars[name] = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_all(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.forward(x)
        if self._features is None:
            raise RuntimeError("Could not capture the backbone pre-logit feature.")
        return {
            "modulation": logits,
            **{name: head(self._features) for name, head in self.aux_heads.items()},
        }

    def embedding_layer(self) -> tuple[str, nn.Linear]:
        _, layer = _last_linear(self.backbone)
        return self._feature_layer_name, layer

    def _capture_features(self, _module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        self._features = inputs[0]


def forward_all(model: nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    if hasattr(model, "forward_all"):
        return model.forward_all(x)  # type: ignore[no-any-return, attr-defined]
    return {"modulation": model(x)}


def _make_head(in_features: int, n_classes: int, hidden_dim: int) -> nn.Module:
    if hidden_dim <= 0:
        return nn.Linear(in_features, n_classes)
    return nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, n_classes),
    )


def _last_linear(model: nn.Module) -> tuple[str, nn.Linear]:
    last = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last = (name, module)
    if last is None:
        raise ValueError("Model has no nn.Linear layer for auxiliary heads.")
    return last
