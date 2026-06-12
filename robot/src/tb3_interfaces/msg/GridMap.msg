# A 2D grid stored row-major as a flat array.
#
# Axis convention (matches the rest of this repo — see PredictAction.srv,
# GridPose.msg, and tb3_planner_common.directions):
#   - The first index ("row") indexes the X axis (East-West, +X = East).
#   - The second index ("col") indexes the Y axis (North-South, +Y = North).
#   - Element at grid cell (x, y):  data[x * cols + y]
#   - "rows" therefore equals the grid's X-extent, "cols" its Y-extent.
#
# This is rotated 90° from the image convention (where rows index Y,
# top-down). Producers must lay out their grid as grid[x][y], NOT grid[y][x].
# A producer that flips this gets a transposed view of the world: planners
# index it as (x, y) but read what's actually at (y, x), so returned action
# directions appear rotated 90° to the consumer.
#
# Used for both obstacle grids (0.0=free, 1.0=occupied) and
# energy grids (floats in [0.0, 1.0]).
# Using float64 for both: the PPO model converts obstacles to int32
# internally via thresholding (> 0), so float64 transport is safe.
uint32 rows
uint32 cols
float64[] data
