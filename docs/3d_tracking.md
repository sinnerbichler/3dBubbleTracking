# 3D bubble tracking and pair statistics

What this does: from the detections of four cameras (`CameraNmidpoints.json`, ~700 bubbles per camera and frame at 1 kHz) it
reconstructs 3D bubble tracks (`src/tracklets.jl`) and computes how the acceleration and the velocity of two bubbles depend on
their distance (`src/statistics.jl`). Figures: `docs/make_figures.jl`.

**Result in one paragraph.** In frames 16 to 135 of `fullrun3_200` the pipeline tracks about 470 bubbles per frame (10 to 90%: 440 to 486),
350 of them with a velocity and 200 with an acceleration (a frame contains ~2800 detections in total, each bubble uses
three or four). The mean radial relative velocity of two bubbles grows with their distance from about 0 at 3 mm to 20 mm/s at 29 mm,
which is the slow widening of the bubble column. The mean radial relative acceleration is compatible with zero at short range
(+0.1 ± 0.2 m/s² at 5 mm, within ±0.3 m/s² between 5 and 11 mm), so no attraction or repulsion between neighbouring bubbles is
detectable at this noise level. Beyond 13 mm there is a small, distance-independent mean of about -0.4 m/s², which is more
likely a large-scale effect or a bias than an interaction.
The share of wrong (ghost) associations is **not measured**, see the caveats.

## 1. Why this is not just triangulation

Each detection is a ray through the flat air/water interface (n = 1.33), and the four rays of one bubble should meet. They do
not, because the acrylic wall bulges outwards under the hydrostatic pressure, by about a bubble diameter. In the matches that are
unambiguous in one frame (4 views, `match_bubbles`) the reprojection error is 5 px in the median and 9 px at the 90th percentile.
The error is a smooth function of the position, but it changes quickly: along some bubble tracks by more than 1 px per frame
(second figure), so it cannot be calibrated away by one constant either.

![refraction](figures/refraction_index.png)

*Distance of the four rays of the 4-view matches of frame 1 from their common point, for different refractive indices of the
water: 5 px at 1.33 and more than 100 px at 1.25 or 1.45. This is an indication that the refraction model works, not a proof:
the matches were found with n = 1.33, which favours it.*

![drift](figures/offset_drift.png)

*Reprojection error of the frame-by-frame 4-view triangulation along eight tracklet groups (one window of 30 frames). On several
groups it changes by 5 to 10 px within 10 to 15 frames, sometimes by more than 1 px per frame. The dashed line is the acceptance
gate of a single frame (`ACCEPT_GATE`). The error contains the noise of the detections as well as the shift.*

With ~700 detections per camera and a nearest-neighbour distance of the bubbles of about 5 mm, a gate that accepts the shifted rays
of the true bubble also accepts the rays of neighbours: at one frame each 2D track of camera 1 had about 25 tracks in camera 2 whose
rays pass within 3 mm. Two views cannot tell the difference, and three or four views only partly. What does not change between the true bubble and a neighbour is that the true
bubble keeps being consistent over time, with the same offsets, while a ghost only fits by accident in single frames.

## 2. The strategies and how far they carry

All numbers are at frame 75 of `fullrun3_200` (2824 detections in the four cameras) unless said otherwise.

| # | strategy | bubbles | evidence | verdict |
|---|---|---|---|---|
| 1 | 2D tracks per camera, then votes for 4-view matches of single frames (`match_tracks`) | ~70 in 50 frames | built from unambiguous raw 4-view matches | precise, but about 1 bubble in 14 |
| 2 | 3D Kalman filter that predicts the bubbles and consumes the detections | ~700 tracks | only 7% of the trusted groups followed, 32% of their detections taken by tracks more than 1.5 mm away | not usable, dropped |
| 3 | tracklet groups (see below), strict consensus, windows of 20 frames, stride 10 | 101 (36 with acceleration) | 1.5% of the detections claimed twice | the most reliable of the tracklet variants, but sparse |
| 4 | tracklet groups without consensus | 790 candidates | between two independent runs only 25 to 39% have a partner within 1.5 mm (3-camera groups 18%, chance 2%) | not reproducible |
| 5 | consensus over many windows (stride 5), windows of 20 or 30, a detection may belong to several bubbles | 483 to 1155 | 13% to 76% of the claimed detections claimed by two or more bubbles | inflated by duplicates |
| **6** | **consensus over many windows, windows of 30, every detection belongs to one bubble** | **475 (193 with acceleration)** with the final knobs, 435 (189) with the narrower ones used before | no detection claimed twice by construction | **used** |

The Kalman filter (2) fails because it associates every camera independently and the shift drifts along the track: a bubble
loses its detections in two cameras, its 3D position is then only constrained along the rays of the others, and it picks up
its neighbours. The tracklet approach (3 to 6) turns this around and first finds out *which* detections belong together, over
many frames, and only then triangulates.

## 3. The pipeline of `src/tracklets.jl`

`track_bubbles(midpoints, theta, frames)` runs steps 1 to 6 and returns the groups that pass the consensus. Step 7 is applied per frame
afterwards (`bubbles_per_frame`).

1. **Tracklets.** In every window of 30 frames the existing 2D tracker `run_tracking` is restarted, so a tracklet is at most 30
   frames long. Short tracklets are what a constant offset per camera can describe, and 2D identity swaps of long tracks are avoided.
2. **Proposals and votes (`frame_tuples`, `associate_tracklets`).** In every second frame all pairs of tracklets of two cameras
   whose rays pass within `DIST_GATE` are triangulated, the tracklets of the other cameras that lie within `SEED_GATE` of the
   reprojection are added, and the proposal is kept if the joint triangulation of all its views reprojects within `ACCEPT_GATE`
   (this is `match_bubbles`, on tracklets and without the greedy exclusivity). A proposal is a group of 3 or 4 tracklets. It has to
   occur in at least `MIN_VOTES` frames.
3. **The fit (`fit_group`).** For each surviving group one trajectory `X(t) = X0 + t V` is fitted to all its rays, with a free
   constant offset `b_c` per camera: the detection of camera `c` is supposed to sit at `X + b_c`. Written as the perpendicular
   distance to each ray (`P = I - n nᵀ`), `P (X0 + t V + b_c - p) = 0`, this is one linear least-squares problem. The offsets
   swallow the wall shift, so only the relative motion has to agree between the cameras and the residual does not depend on how
   far the whole point cloud is shifted. A weak prior (`SIGMA_OFFSET_PX`) on the offsets removes the ambiguity between the
   offsets and `X0`. Groups with a residual above `MAX_RMS` or fewer than `MIN_FRAMES` common frames are dropped.
4. **Selection.** Groups are ranked by the number of cameras, the number of frames in which the proposal occurred (not their
   fraction: the shift drifts beyond the gates in parts of a long track), and the residual. Best first, a group is accepted if
   none of its tracklets is used yet.
5. **Raw detections (`track_window`).** The 2D Kalman filter that produced the tracklets lags behind accelerations, so the
   accepted groups are refitted and triangulated through the raw detection closest to each filtered point (within `RAW_RADIUS`;
   for coasting frames the filtered point stays). The positions are triangulated frame by frame with the fitted offsets removed.
6. **Consensus (`consensus_filter`).** Windows of 30 frames start every 5 frames, so every frame is in six windows with differently
   cut tracklets. A group is kept if another window reproduces it: at least half of the frames that can be compared (at least 5)
   have a group of the other window within `CONSENSUS_DIST`. Frames closer than `CONSENSUS_MARGIN` to the ends of the other window
   are not compared, tracklets are cut there and groups are scarce, so a real bubble would fail to reproduce for no reason.
   A ghost is a particular combination of tracklets, and other windows rarely produce the same combination.
7. **Exclusive bubbles per frame (`exclusive_bubbles_at`).** The same bubble can be in several groups with positions a few mm
   apart. For every frame the groups are ranked (more cameras, longer), each claims the detection closest to its reprojection
   in each of its cameras, and a group whose detection is already claimed is dropped. Nothing else stops the window overlap from
   double counting: allowing two claims per detection roughly doubles the bubbles, but 68% of the detections are then used
   twice.

The knobs were widened a little relative to the values used during development (`DIST_GATE` 3 to 4 mm, `SEED_GATE` 15 to 18 px,
`ACCEPT_GATE` 10 to 12 px, `MAX_RMS` 15 to 18 px, `CONSENSUS_DIST` 1.5 to 2 mm, `CONSENSUS_FRAC` 0.6 to 0.5). A larger gate
means more candidates, and the consensus and the exclusivity stay the filters that keep the ghosts out. The effect was small
(median bubbles per frame 434 to 470, with acceleration 189 to 200), because the exclusivity caps what can be added.

![yield](figures/yield.png)

*Tracked bubbles per frame. The velocity curve has a sawtooth with the period of the window stride (5 frames), which is not physical. The yield falls near
the ends of the range because fewer windows cover those frames.*

## 4. What it gives

Over frames 16 to 135, median per frame: **470 bubbles, 352 with a velocity, 200 with an acceleration**. About 60% of the
detections of a frame end up in a bubble (at frame 75: 1696 of 2824). With the 4-camera groups only, the yield is 290 bubbles and
129 with an acceleration. 13,646 groups pass the consensus (60% of them 4-camera), with a median length of 15 frames (10%: 9, 90%: 30;
12% fill the whole window). The median residual of the trajectory fit is 3.5 px, the residual of the local parabola of a bubble
is 35 µm. Nothing is stitched across windows, so a track is at most 30 frames long.

At frame 75, 475 bubbles exist. 391 of them have a track of at least two frames between frame 75 and 85, 226 of these are 4-camera
groups, and 194 run through all 10 frames.

![cameras](figures/tracks_cameras.png)

*Tracks of the bubbles of frame 75 up to frame 85 reprojected into the four cameras (red: 4-camera groups, magenta: 3-camera
groups). There is no image for these frames, so the background is the raw detections of frame 75 (blue) and of frame 85 (orange).
A good track starts on a blue and ends on an orange point. This shows that the tracks are consistent with the detections, it does
not show that the detection of each camera belongs to the same bubble.*

![3d](figures/tracks_3d.png)

*The same tracks in 3D (at least 6 frames). All tracked bubbles are shown, including the dense region at the bottom near the sparger
(z below -80 mm), which is excluded from the statistics.*

## 5. Statistics (`src/statistics.jl`)

**Kinematics.** `local_kinematics` fits a parabola to the 11 positions around a frame (`half=5`, 1 ms per frame) of a group.
Position, velocity and acceleration are the coefficients. The rms of the fit is about 35 µm, which gives an acceleration noise of
about 2 to 3 m/s² per bubble and component. It is never fitted across two groups: their positions differ by a constant (the gauge of the offsets), and
a jump of 0.5 mm inside the 11 frames of a fit gives an acceleration error of 6 to 25 m/s², depending on where it falls.

**Pairs.** `pair_statistics` takes all pairs of bubbles of a frame closer than 30 mm and averages, in bins of the distance `r`,
the radial component of the relative acceleration, `(a_j - a_i) · r̂`, and of the relative velocity, `(v_j - v_i) · r̂`,
with `r̂` pointing from `i` to `j`. Positive means the two bubbles accelerate or move apart. The result does not depend on which
bubble is `i`.

**Uncertainty.** The same pair shows up in many consecutive frames, and every fit spans 11 frames, so the pairs are far from
independent. The error bars are the standard deviation of a bootstrap over blocks of 10 frames. With 12 blocks they are themselves
rough.

![statistics](figures/pair_statistics.png)

| distance (mm) | pairs | acceleration, all groups (m/s²) | acceleration, 4-camera only (m/s²) | relative velocity, all groups (mm/s) |
|---|---|---|---|---|
| 3 | 1292 | -0.64 ± 0.38 | -0.33 ± 0.41 | -0.1 ± 4.3 |
| 5 | 4448 | 0.13 ± 0.22 | 0.70 ± 0.26 | 2.3 ± 1.5 |
| 7 | 8148 | 0.28 ± 0.15 | 0.24 ± 0.18 | 5.7 ± 1.4 |
| 11 | 18788 | 0.04 ± 0.09 | 0.23 ± 0.10 | 10.2 ± 1.4 |
| 15 | 31012 | -0.35 ± 0.07 | -0.29 ± 0.10 | 12.0 ± 1.1 |
| 19 | 42189 | -0.29 ± 0.08 | -0.35 ± 0.10 | 16.2 ± 0.7 |
| 25 | 53534 | -0.42 ± 0.10 | -0.47 ± 0.13 | 18.5 ± 0.9 |
| 29 | 56111 | -0.37 ± 0.08 | -0.42 ± 0.13 | 20.4 ± 1.4 |

443,044 pairs (198,165 with the 4-camera groups only) in frames 16 to 135, block bootstrap over 12 blocks. In the figure, blue is all groups and
orange the 4-camera subset.

- **Relative velocity.** It grows from about 0 at 3 to 5 mm (bubbles that close move together) to 20 mm/s at 29 mm. This is the
  widening of the bubble column: the further two bubbles are apart, the more they move away from each other on average.
- **Relative acceleration at short range.** Between 5 and 11 mm it is within ±0.3 m/s² of zero, with thousands of pairs per bin
  (the noise of the radial relative acceleration of one pair is about 3 to 4 m/s²). At 3 mm the result is -0.6 ± 0.4, which is not significant, and the number of pairs is small.
  The 4-camera subset has +0.7 ± 0.3 at 5 mm, which the full set does not show (0.13 ± 0.22). The bins are neighbours in a
  single measurement, not independent tests, so this is not read as an effect.
- **Beyond 13 mm** the mean is about -0.3 to -0.5 m/s² in both sets. It does not depend on the distance, which is not what a
  short-range interaction looks like. It may be a large-scale acceleration gradient of the flow or a bias of the reconstruction.
- **Sensitivity.** At 5 to 11 mm the bootstrap uncertainty is about ±0.1 to 0.3 m/s²; at 3 mm it is about ±0.4 m/s² and at 1 mm
  more than ±1 m/s². A mean interaction acceleration much larger than that would show up, subject to the caveats below.

## 6. Caveats

- **Ghost rate not measured.** A ghost is built from real detections of different bubbles, so overlays such as the one above look
  fine. The only proxies are the detection exclusivity (no detection is used twice) and, in an early variant, the reproducibility
  between independent runs (4-camera groups 35 to 43% had a partner, 3-camera groups 18%). A manual sample of ~40 groups would give a
  number with an error bar.
- **The consensus is not independent:** all windows use the same detections. A ghost from a geometry that stays the same could be
  reproduced.
- **Only frames 1 to 150** of one run (`fullrun3_200`) were processed, images exist only for frames 1 to 3.
- **Positions carry a gauge:** the offsets are only fixed up to the weak prior, so positions of two groups of the same bubble differ by
  up to a millimetre. Distances between bubbles of different groups inherit this.
- **`DOMAIN_LO`/`DOMAIN_HI`** (the volume of the statistics, in particular the lower bound in z against the sparger) are a guess, not
  a measured boundary.
- **The refraction check is biased** (see the first figure).
- **An exclusive selection may drop real bubbles** that share a detection because they overlap in one camera.
- The error bars of the statistics come from 12 blocks.

## 7. How to run

```julia
using GLMakie # tracking.jl defines methods on GLMakie.lines!
include("src/statistics.jl")

# theta as in calibration.jl:  sol = run_calibration(detections_list, intersections_list) ...  theta = merge_free_and_fixed_parameters(sol.u, fixed_parameters)
casepath = "/home/simon/mega/masterarbeit/fullrun3_200/"
midpoints_per_camera_per_frame = load_midpoints(casepath .* ["Camera$(c)midpoints.json" for c in 1:4])

groups, windows = track_bubbles(midpoints_per_camera_per_frame, theta, 1:150)       # about 10 minutes
bubbles = bubbles_per_frame(groups, midpoints_per_camera_per_frame, theta, 16:135)   # every frame is in at least four windows here
statistics = pair_statistics(bubbles, 16:135)
fig = plot_pair_statistics(statistics)
```

The result of the run described here is stored in `generated-files/` (git ignored, `Serialization`, Julia 1.12.7):
`tracklet_state_frames1-150.jls` (`groups`, `windows`, `theta`) and `tracklet_results_frames1-150.jls` (yield and statistics).
`julia --project=. docs/make_figures.jl` regenerates the figures from them.
