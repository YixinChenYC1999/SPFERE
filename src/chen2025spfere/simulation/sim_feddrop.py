from typing import Tuple
from torch import nn

_CONV_TYPES = (
    nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d
)

def _unwrap_parallel(model: nn.Module) -> nn.Module:
    return getattr(model, "module", model)

def count_conv_fc_params(model: nn.Module) -> Tuple[int, int, int, float, float]:
    model = _unwrap_parallel(model)

    conv_params = 0
    fc_params = 0

    for m in model.modules():
        if isinstance(m, _CONV_TYPES):
            conv_params += sum(p.numel() for p in m.parameters() if p.requires_grad)
        elif isinstance(m, nn.Linear):
            fc_params += sum(p.numel() for p in m.parameters() if p.requires_grad)

    total = conv_params + fc_params
    other_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ) - total
    total_all = total + other_trainable

    if total_all > 0:
        alpha_conv = conv_params / total_all
        alpha_fc   = fc_params / total_all
    else:
        alpha_conv = alpha_fc = 0.0

    return alpha_conv, alpha_fc