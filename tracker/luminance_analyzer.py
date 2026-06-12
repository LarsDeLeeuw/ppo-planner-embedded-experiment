"""
luminance_analyzer.py — Compute per-cell average luminance from a warped grid image.

Implements the GridAnalyzer protocol.  Port of src/energy-map/luminance.py
from PIL to OpenCV so the entire tracker pipeline stays in one imaging library.
"""

from __future__ import annotations

import cv2
import numpy as np

from analyzer import AnalysisResult


class LuminanceAnalyzer:
    """Average grayscale luminance per grid cell, normalized to [0, 1]."""

    name = "luminance"

    def analyze(
        self, image: np.ndarray, rows: int, cols: int,
    ) -> AnalysisResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        cell_h, cell_w = h // rows, w // cols

        grid = np.zeros((rows, cols), dtype=np.float64)
        for r in range(rows):
            for c in range(cols):
                cell = gray[r * cell_h : (r + 1) * cell_h,
                            c * cell_w : (c + 1) * cell_w]
                grid[r, c] = cell.mean()

        max_val = grid.max()
        if max_val > 0:
            grid /= max_val

        overlay = self._draw_overlay(image.copy(), grid, rows, cols)

        return AnalysisResult(
            grid=grid,
            overlay=overlay,
            metadata={"unit": "relative_luminance", "range": [0.0, 1.0]},
        )

    @staticmethod
    def _draw_overlay(
        image: np.ndarray,
        grid: np.ndarray,
        rows: int,
        cols: int,
    ) -> np.ndarray:
        """Draw grid lines and per-cell luminance values on the image."""
        h, w = image.shape[:2]
        cell_h, cell_w = h // rows, w // cols

        # Grid lines
        for r in range(1, rows):
            y = r * cell_h
            cv2.line(image, (0, y), (w, y), (255, 255, 255), 2)
        for c in range(1, cols):
            x = c * cell_w
            cv2.line(image, (x, 0), (x, h), (255, 255, 255), 2)

        # Border
        cv2.rectangle(image, (0, 0), (w - 1, h - 1), (255, 255, 255), 2)

        # Cell values
        font = cv2.FONT_HERSHEY_SIMPLEX
        for r in range(rows):
            for c in range(cols):
                val = grid[r, c]
                cx = c * cell_w + cell_w // 2
                cy = r * cell_h + cell_h // 2
                text = f"{val:.2f}"
                (tw, th), _ = cv2.getTextSize(text, font, 0.6, 2)
                cv2.putText(
                    image, text,
                    (cx - tw // 2, cy + th // 2),
                    font, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
                )

        return image
