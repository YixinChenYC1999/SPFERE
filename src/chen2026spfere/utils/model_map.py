from chen2026spfere.utils.defined_models import SimpleCNN, SimpleCNN_FM, ShuffleCNN, EfficientCNN, EfficientCNN_DP
from chen2026spfere.utils.resnet import resnet20

MODEL_MAP = {
    "SimpleCNN": SimpleCNN,
    "ShuffleCNN": ShuffleCNN,
    "EfficientCNN": EfficientCNN,
    "resnet20": resnet20,
    "SimpleCNN_FM": SimpleCNN_FM,
    "EfficientCNN_DP": EfficientCNN_DP,
}