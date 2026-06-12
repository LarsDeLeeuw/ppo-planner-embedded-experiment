"""
capture_pipeline.py — Orchestrates grid image capture, perspective warp, and
pluggable analysis.

Usage::

    from capture_pipeline import CapturePipeline
    from luminance_analyzer import LuminanceAnalyzer

    pipeline = CapturePipeline("captures", warp_size=800)
    pipeline.register(LuminanceAnalyzer())

    # When the user triggers a capture:
    results = pipeline.run(raw_frame, grid)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from analyzer import AnalysisResult, GridAnalyzer
from grid import GridState
from perspective import warp_to_square


class CapturePipeline:
    """Register analyzers, capture a frame, warp it, run all analyzers, save."""

    def __init__(self, output_dir: str, warp_size: int) -> None:
        self._output_dir = Path(output_dir)
        self._warp_size = warp_size
        self._analyzers: list[GridAnalyzer] = []

    def register(self, analyzer: GridAnalyzer) -> None:
        """Add an analyzer to the pipeline."""
        self._analyzers.append(analyzer)

    def run(
        self, frame: np.ndarray, grid: GridState,
    ) -> dict[str, AnalysisResult]:
        """Execute the full capture pipeline.

        1. Save raw frame
        2. Perspective-warp to square
        3. Save warped image
        4. Run every registered analyzer
        5. Save each overlay
        6. Write metadata sidecar

        Returns a dict mapping analyzer name → AnalysisResult.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self._output_dir / ts
        run_dir.mkdir(parents=True, exist_ok=True)

        # --- Raw ---
        raw_path = str(run_dir / "raw.png")
        cv2.imwrite(raw_path, frame)
        print(f"[capture] saved {raw_path}")

        # --- Warp ---
        warped, M = warp_to_square(frame, grid.src_pts, self._warp_size)
        warped_path = str(run_dir / "warped.png")
        cv2.imwrite(warped_path, warped)
        print(f"[capture] saved {warped_path}")

        # --- Analyze ---
        results: dict[str, AnalysisResult] = {}
        for analyzer in self._analyzers:
            result = analyzer.analyze(warped, grid.rows, grid.cols)
            results[analyzer.name] = result
            if result.overlay is not None:
                overlay_path = str(run_dir / f"{analyzer.name}.png")
                cv2.imwrite(overlay_path, result.overlay)
                print(f"[capture] saved {overlay_path}")

            np.set_printoptions(precision=3, suppress=True)
            print(f"[capture] {analyzer.name} grid:\n{result.grid}")

        # --- Metadata sidecar ---
        meta = {
            "timestamp": ts,
            "grid": {"rows": grid.rows, "cols": grid.cols},
            "warp_size": self._warp_size,
            "analyzers": {
                name: {
                    **r.metadata,
                    "grid_values": r.grid.tolist(),
                }
                for name, r in results.items()
            },
        }
        meta_path = run_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[capture] saved {meta_path}")

        return results
