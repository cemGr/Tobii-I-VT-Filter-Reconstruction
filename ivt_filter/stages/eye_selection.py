"""Eye selection stage (Olsen Section 3.1.2)."""
from __future__ import annotations

from statistics import mean

from .base import IFilterStage
from ..config import EyeSelectionMode, IVTFilterConfiguration
from ..domain.dataset import Recording


class EyeSelectionStage(IFilterStage):
    """Compute combined gaze according to the configured strategy."""

    def process(self, recording: Recording, config: IVTFilterConfiguration) -> None:
        mode = config.eye_selection_mode
        for sample in recording.samples:
            left_ok = sample.is_left_valid()
            right_ok = sample.is_right_valid()
            lx, ly = sample.left_eye.gaze_x, sample.left_eye.gaze_y
            rx, ry = sample.right_eye.gaze_x, sample.right_eye.gaze_y
            combined_valid = False
            combined_x = None
            combined_y = None
            if mode == EyeSelectionMode.LEFT:
                if left_ok:
                    combined_valid = True
                    combined_x, combined_y = lx, ly
            elif mode == EyeSelectionMode.RIGHT:
                if right_ok:
                    combined_valid = True
                    combined_x, combined_y = rx, ry
            elif mode == EyeSelectionMode.AVERAGE:
                if left_ok and right_ok:
                    combined_valid = True
                    combined_x = mean([lx, rx])
                    combined_y = mean([ly, ry])
                elif left_ok:
                    combined_valid = True
                    combined_x, combined_y = lx, ly
                elif right_ok:
                    combined_valid = True
                    combined_x, combined_y = rx, ry
            elif mode == EyeSelectionMode.STRICT_AVERAGE:
                if left_ok and right_ok:
                    combined_valid = True
                    combined_x = mean([lx, rx])
                    combined_y = mean([ly, ry])
            sample.combined_gaze_x = combined_x
            sample.combined_gaze_y = combined_y
            sample.combined_valid = combined_valid
