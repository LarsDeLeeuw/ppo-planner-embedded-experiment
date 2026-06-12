"""
analyzer.py — Grid analyzer protocol and result type.

Defines the contract that any pluggable grid analyzer must satisfy so the
capture pipeline can orchestrate it without knowing its internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class AnalysisResult:
    """Output of a single analyzer run."""
    grid: np.ndarray                            # (rows, cols) numeric result
    overlay: np.ndarray | None                  # annotated image, or None if headless
    metadata: dict[str, Any] = field(default_factory=dict)


class GridAnalyzer(Protocol):
    """Interface every grid analyzer must implement."""
    name: str

    def analyze(
        self, image: np.ndarray, rows: int, cols: int,
    ) -> AnalysisResult:
        """Analyze a perspective-corrected grid image.

        Parameters
        ----------
        image : np.ndarray
            (H, W, 3) BGR square image of the warped grid region.
        rows : int
            Number of grid rows.
        cols : int
            Number of grid columns.

        Returns
        -------
        AnalysisResult
            Numeric grid, optional overlay, and freeform metadata.
        """
        ...
