"""AMBER: Adaptive Multimodal Mask Transformer for beam prediction.

Implementation of Wen et al., for DeepSense6G scenarios 31-34.
"""
from .config import MODALITIES, ModelConfig, TrainConfig
from .model import AMBER, AmberOutput

__all__ = ["AMBER", "AmberOutput", "ModelConfig", "TrainConfig", "MODALITIES"]
