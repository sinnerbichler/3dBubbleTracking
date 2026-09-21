# code to reconstruct 3d bubble tracks from the four cameras
#
# the rays of one bubble do not meet in a single frame: the wall shift (acrylic wall bulging out under the hydrostatic pressure)
# misses them by up to a bubble diameter, and this error changes along a track. instead of triangulating frame by frame, short
# 2d tracks (tracklets) of the four cameras are associated over ~30 frames, using a model that is insensitive to the shift:
#   1. run_tracking (tracking.jl) gives 2d tracklets per camera
#   2. frame_tuples proposes groups of 3 or 4 tracklets that fit together in one frame, the proposals are voted over the frames
#   3. fit_group fits one linear 3d trajectory to a group with a free constant offset per camera (the shift)
#   4. associate_tracklets keeps the best groups, every tracklet is used once
#   5. track_windows repeats this in overlapping windows, consensus_filter keeps the groups that another window reproduces
#   6. exclusive_bubbles_at lets every detection belong to one bubble only, per frame
# see docs/3d_tracking.md for the reasoning behind the steps

using StaticArrays
using LinearAlgebra
using Statistics
using NearestNeighbors

include("tracking.jl")

# knobs, distances in metres and pixels in image pixels
const DIST_GATE = 4e-3          # maximum distance of two rays to propose them as a pair
const SEED_GATE = 18.0f0        # px, a third/fourth view has to be this close to the reprojected pair
const ACCEPT_GATE = 12.0f0      # px, maximum reprojection error of the joint fit of a proposal
const MIN_VOTES = 3             # frames (of every second one) in which a proposal has to occur
const MIN_FRAMES = 8            # minimum number of common frames of a group
const MAX_RMS = 18.0            # px, maximum residual of the trajectory fit of a group
const SIGMA_PX = 2.0            # px, noise of a detection in the fit
const SIGMA_OFFSET_PX = 10.0    # px, prior on the offset of a camera, it also fixes the gauge between offset and position
const CONSENSUS_DIST = 2e-3     # a group is reproduced if another window has a group this close ...
const CONSENSUS_FRAC = 0.5      # ... in this fraction of the frames ...
const CONSENSUS_MIN_FRAMES = 5  # ... of at least this many frames that can be compared
const CONSENSUS_MARGIN = 3      # frames at the ends of a window are not used for comparing, tracklets are cut there
const CLAIM_RADIUS = 12.0f0     # px, a bubble claims the detection closest to its reprojection within this radius
const RAW_RADIUS = 4.0f0        # px, a filtered 2d track point is replaced by the raw detection within this radius
const PX_PER_M = 1.7e4          # fx / 0.7 m, only used to weight the fit and to convert its residual to pixels

# rays through the (kalman filtered) history points of the tracklets, rays[c][id][k] belongs to frame start_frame + k - 1
function tracklet_rays(tracks_per_camera, theta)
    return [Dict(id => [waterray_from_camera(point..., theta, c, 1.0, 1.33) for point in track.history] for (id, track) in tracks_per_camera[c])
            for c in eachindex(tracks_per_camera)]
end

# the same, but through the raw detection closest to the history point (the 2d kalman filter lags behind accelerations),
# only for the tracklets that are part of the groups. if there is none, the history point is used
function raw_tracklet_rays(groups, tracks_per_camera, midpoints_per_camera_per_frame, theta, window)
    rays = [Dict{Int, Vector{Ray{Float64}}}() for _ in 1:4]
    for group in groups, c in 1:4
        id = group.ids[c]
        (id == 0 || haskey(rays[c], id)) && continue
        track = tracks_per_camera[c][id]
        rays[c][id] = map(enumerate(track.history)) do (k, point)
            detections = midpoints_per_camera_per_frame[c][first(window) - 1 + track.start_frame + k - 1]
            distance, index = findmin([norm(detection - point) for detection in detections])
            return waterray_from_camera((distance < RAW_RADIUS ? detections[index] : point)..., theta, c, 1.0, 1.33)
        end
    end
    return rays
end

# index[c][f] = (id, point) of the tracklets that are alive in camera c at frame f
function tracklet_frame_index(tracks_per_camera, nframes)
    index = [[Tuple{Int, SVector{2, Float32}}[] for _ in 1:nframes] for _ in eachindex(tracks_per_camera)]
    for (c, tracks) in enumerate(tracks_per_camera), (id, track) in tracks, (k, point) in enumerate(track.history)
        f = track.start_frame + k - 1
        f <= nframes && push!(index[c][f], (id, point))
    end
    return index
end

# proposals of one frame, as tuples of tracklet ids (0 = camera not in the tuple). same seeding and support logic as
# match_bubbles, but on the tracklets and without the greedy exclusivity: ambiguous proposals survive until they are judged over time
function frame_tuples(index, rays, tracks_per_camera, theta, f; dist_gate=DIST_GATE, seed_gate=SEED_GATE, accept_gate=ACCEPT_GATE, min_views=3)
    ncams = length(index)
    ids = [first.(index[c][f]) for c in 1:ncams]
    points = [last.(index[c][f]) for c in 1:ncams]
    rays_f = [[rays[c][id][f - tracks_per_camera[c][id].start_frame + 1] for id in ids[c]] for c in 1:ncams]
    trees = [isempty(points[c]) ? nothing : KDTree(points[c]) for c in 1:ncams]

    proposals = Set{NTuple{4, Int}}()
    for a in 1:ncams, b in a+1:ncams, i in eachindex(rays_f[a]), j in eachindex(rays_f[b])
        p3d, d = mean_point_and_distance(rays_f[a][i], rays_f[b][j])
        abs(d) > dist_gate && continue

        tup = zeros(Int, ncams)
        tup[a] = i; tup[b] = j
        for c in setdiff(1:ncams, (a, b))
            isnothing(trees[c]) && continue
            idx, dist = knn(trees[c], project_point_onto_image_plane(p3d, c, theta), 1)
            dist[1] < seed_gate && (tup[c] = idx[1])
        end
        count(!=(0), tup) < min_views && continue

        used_cams = findall(!=(0), tup)
        refined = triangulate_rays([rays_f[c][tup[c]] for c in used_cams])
        err = maximum(norm(project_point_onto_image_plane(refined, c, theta) - points[c][tup[c]]) for c in used_cams)
        err > accept_gate && continue

        push!(proposals, ntuple(c -> tup[c] == 0 ? 0 : ids[c][tup[c]], ncams))
    end
    return proposals
end

# fits X(t) = X0 + t*V to the rays of a group, with a constant offset b_c per camera: the detection of camera c is
# supposed to sit at X + b_c. the offsets take up the wall shift, so only the relative motion enters the residual.
# linear least squares in (X0, V, b_1, ..., b_m), the ridge on the offsets removes the degeneracy between X0 and the offsets.
# tracklets[c] is a track or nothing. returns nothing if fewer than 3 cameras or MIN_FRAMES common frames
function fit_group(tracklets, rays, ids, theta; min_frames=MIN_FRAMES)
    cams = findall(!isnothing, tracklets)
    length(cams) >= 3 || return nothing
    ranges = [tracklets[c].start_frame:(tracklets[c].start_frame + length(tracklets[c].history) - 1) for c in cams]
    frames = reduce(intersect, ranges)
    length(frames) >= min_frames || return nothing
    tmid = (first(frames) + last(frames)) / 2

    m = length(cams)
    A = zeros(6 + 3m, 6 + 3m)
    b = zeros(6 + 3m)
    w = (PX_PER_M / SIGMA_PX)^2
    observations = Tuple{Int, Float64, SMatrix{3, 3, Float64, 9}, SVector{3, Float64}}[]
    for (ci, c) in enumerate(cams), f in frames
        ray = rays[c][ids[c]][f - tracklets[c].start_frame + 1]
        push!(observations, (ci, f - tmid, SMatrix{3, 3, Float64}(I) - ray.n * ray.n', SVector{3, Float64}(ray.p))) # projector normal to the ray
    end
    for (ci, t, P, p) in observations
        offset = 6 + 3ci - 2
        for (i, gi) in enumerate((1.0, t)), (j, gj) in enumerate((1.0, t))
            A[3i-2:3i, 3j-2:3j] .+= w * gi * gj * P
        end
        for (i, gi) in enumerate((1.0, t))
            A[3i-2:3i, offset:offset+2] .+= w * gi * P
            A[offset:offset+2, 3i-2:3i] .+= w * gi * P
            b[3i-2:3i] .+= w * gi * (P * p)
        end
        A[offset:offset+2, offset:offset+2] .+= w * P
        b[offset:offset+2] .+= w * (P * p)
    end
    for ci in 1:m
        offset = 6 + 3ci - 2
        A[offset:offset+2, offset:offset+2] .+= (PX_PER_M / SIGMA_OFFSET_PX)^2 * Matrix(I, 3, 3)
    end
    u = A \ b

    X0, V = SVector{3}(u[1:3]), SVector{3}(u[4:6])
    offsets = [SVector{3}(u[6+3ci-2:6+3ci]) for ci in 1:m]
    residual = sum(sum(abs2, P * (X0 + t * V + offsets[ci] - p)) for (ci, t, P, p) in observations)
    return (rms=sqrt(residual / (2 * length(observations))) * PX_PER_M, X0=X0, V=V, offsets=offsets, frames=frames, cams=cams)
end

# position of a fitted group in every frame: triangulation of the rays with the fitted offsets removed
# (the fit itself is linear in time, this keeps the accelerations)
function group_positions(fit, tracklets, rays, ids)
    return map(fit.frames) do f
        shifted = [begin
                       ray = rays[c][ids[c]][f - tracklets[c].start_frame + 1]
                       Ray{Float64}(ray.n, ray.p - fit.offsets[ci])
                   end for (ci, c) in enumerate(fit.cams)]
        SVector{3, Float64}(triangulate_rays(shifted))
    end
end

# votes of the proposals over the frames -> fits -> greedy exclusive selection.
# ranking: number of cameras, then the number of frames in which the proposal occurred, then the residual of the fit
# (the fraction of frames is misleading, the shift drifts beyond the gates in parts of a track)
function associate_tracklets(tracks_per_camera, theta, frames)
    nframes = maximum(track.start_frame + length(track.history) - 1 for tracks in tracks_per_camera for track in values(tracks))
    index = tracklet_frame_index(tracks_per_camera, nframes)
    rays = tracklet_rays(tracks_per_camera, theta)

    votes = Dict{NTuple{4, Int}, Int}()
    for f in first(frames):2:last(frames)
        for tup in frame_tuples(index, rays, tracks_per_camera, theta, f)
            votes[tup] = get(votes, tup, 0) + 1
        end
    end

    candidates = Tuple{NTuple{4, Int}, Any}[]
    for (ids, v) in votes
        v >= MIN_VOTES || continue
        tracklets = Union{Nothing, Track}[ids[c] == 0 ? nothing : tracks_per_camera[c][ids[c]] for c in 1:4]
        fit = fit_group(tracklets, rays, ids, theta)
        (isnothing(fit) || fit.rms > MAX_RMS) && continue
        push!(candidates, (ids, fit))
    end
    sort!(candidates, by = c -> (-length(c[2].cams), -votes[c[1]], c[2].rms))

    used = [Set{Int}() for _ in 1:4]
    groups = NamedTuple[]
    for (ids, fit) in candidates
        any(c -> ids[c] != 0 && ids[c] in used[c], 1:4) && continue
        for c in 1:4
            ids[c] != 0 && push!(used[c], ids[c])
        end
        push!(groups, (ids=ids, fit=fit))
    end
    return groups
end

# the 3d groups of one window of frames: 2d tracking restarted in the window (so tracklets are at most as long as the window),
# association on the filtered tracklets, then refit and triangulation with the raw detections.
# returns groups with ncams, fit, positions[k] (belongs to frames[k]) and frames in global frame numbers
function track_window(midpoints_per_camera_per_frame, theta, window::UnitRange)
    midpoints = [midpoints_per_camera_per_frame[c][window] for c in 1:4]
    tracks_per_camera = [run_tracking(midpoints[c]) for c in 1:4]
    tracks_per_camera = [filter!(p -> p.second.hits > 4 && length(p.second.history) > 4, tracks) for tracks in tracks_per_camera]

    groups = associate_tracklets(tracks_per_camera, theta, 1:length(window))
    rays = raw_tracklet_rays(groups, tracks_per_camera, midpoints_per_camera_per_frame, theta, window)
    refined = NamedTuple[]
    for group in groups
        tracklets = Union{Nothing, Track}[group.ids[c] == 0 ? nothing : tracks_per_camera[c][group.ids[c]] for c in 1:4]
        fit = fit_group(tracklets, rays, group.ids, theta)
        isnothing(fit) && continue
        push!(refined, (ncams=length(fit.cams), fit=fit, positions=group_positions(fit, tracklets, rays, group.ids),
                        frames=fit.frames .+ (first(window) - 1)))
    end
    return refined
end

# all windows of `width` frames with the given stride: 30 frames give clearly more usable tracks than 20 (a third of the groups
# hit the window length), the stride of 5 puts every frame in six windows, which the consensus needs
function track_windows(midpoints_per_camera_per_frame, theta, frames; width=30, stride=5)
    windows = [s:(s + width - 1) for s in first(frames):stride:(last(frames) - width + 1)]
    return [track_window(midpoints_per_camera_per_frame, theta, window) for window in windows], windows
end

# keeps the groups that other windows reproduce: a frame of a group is compared if another window covers it away from
# its ends (there tracklets are cut and groups are scarce), and counts as reproduced if that window has a group close by
function consensus_filter(window_groups, windows)
    trees = [Dict{Int, KDTree}() for _ in windows]
    for (i, groups) in enumerate(window_groups), f in windows[i]
        points = [g.positions[f - first(g.frames) + 1] for g in groups if f in g.frames]
        isempty(points) || (trees[i][f] = KDTree(reduce(hcat, Vector.(points))))
    end

    kept = NamedTuple[]
    for (i, groups) in enumerate(window_groups), group in groups
        compared = 0
        reproduced = 0
        for f in group.frames
            candidates = [j for j in eachindex(windows) if j != i && f in windows[j] && haskey(trees[j], f) &&
                          f - first(windows[j]) >= CONSENSUS_MARGIN && last(windows[j]) - f >= CONSENSUS_MARGIN]
            isempty(candidates) && continue
            compared += 1
            point = Vector(group.positions[f - first(group.frames) + 1])
            any(knn(trees[j][f], point, 1)[2][1] < CONSENSUS_DIST for j in candidates) && (reproduced += 1)
        end
        (compared >= CONSENSUS_MIN_FRAMES && reproduced / compared >= CONSENSUS_FRAC) && push!(kept, group)
    end
    return kept
end

# the bubbles of one frame: (group, position) pairs. windows overlap, so a bubble can be in several groups with positions a few mm
# apart (different offsets). groups are ranked (more cameras, longer first), every group claims the detection closest to its
# reprojection in each camera, and a group is dropped if one of them is already claimed by a better group
function exclusive_bubbles_at(groups, f, midpoints_per_camera_per_frame, theta)
    candidates = [(g, g.positions[f - first(g.frames) + 1]) for g in groups if f in g.frames]
    sort!(candidates, by = c -> (-c[1].ncams, -length(c[1].frames)))

    claimed = [Set{Int}() for _ in 1:4]
    bubbles = typeof(candidates)()
    for (group, position) in candidates
        mine = Tuple{Int, Int}[]
        for (ci, c) in enumerate(group.fit.cams)
            projected = project_point_onto_image_plane(SVector{3, Float64}(position + group.fit.offsets[ci]), c, theta)
            distance, index = findmin([norm(detection - projected) for detection in midpoints_per_camera_per_frame[c][f]])
            distance < CLAIM_RADIUS && push!(mine, (c, index))
        end
        any(m -> m[2] in claimed[m[1]], mine) && continue
        for (c, index) in mine
            push!(claimed[c], index)
        end
        push!(bubbles, (group, position))
    end
    return bubbles
end

# tracks the bubbles in the given frames: groups that are reproduced by the overlapping windows
function track_bubbles(midpoints_per_camera_per_frame, theta, frames; width=30, stride=5)
    window_groups, windows = track_windows(midpoints_per_camera_per_frame, theta, frames; width, stride)
    return consensus_filter(window_groups, windows), windows
end
