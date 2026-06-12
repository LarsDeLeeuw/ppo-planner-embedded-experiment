# experiments

Campaign definitions and the **derived** outputs of experiment runs — figures, per-run summaries
(`summary.json` / `summary.png`), metadata, and small artifacts.

> **Raw bags are not committed.** The heavy ROS `.mcap` recordings (hundreds of MB) are kept out of
> the repo (`*.mcap` is git-ignored) and offered as a separate download. This directory keeps only
> the figures and summaries needed to read the results. *(Population of this directory is tracked as
> Chunk 4 of the consolidation — see [docs/workflows](../docs/workflows/README.md).)*

Generate or refresh outputs with the tools in [`../analysis/`](../analysis/). Each run folder
(`runs/<run_id>/`) holds the planner/mode/map it was produced under; see
[../docs/running.md](../docs/running.md).
