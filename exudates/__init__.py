"""Exudate detection in color fundus images by adaptive color morphology."""
from .detection import Params, Result, detect_exudates, overlay

__all__ = ["Params", "Result", "detect_exudates", "overlay"]
