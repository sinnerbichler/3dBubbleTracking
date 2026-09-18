# code to track bubbles across the image planes
# kalman filter implementation for constant velocity model

using StaticArrays
using LinearAlgebra
using SparseArrays
using Hungarian
using JSON
using GLMakie
using ImageIO
using FileIO
using NearestNeighbors
using ProgressBars
using Statistics

include("calibartion.jl")

const MAX_MISSES = 3
const GATE_DIST = 15 # px

# datatypes

mutable struct KalmanFilter
    x::MVector{4, Float32}
    P::MMatrix{4, 4, Float32, 16}
end

mutable struct Track
    # id::Int # id is given by the key that points to it in tracks
    kf::KalmanFilter
    start_frame::Int
    last_seen::Int
    hits::Int
    misses::Int
    history::Vector{SVector{2, Float32}}
end

const F::SMatrix{4, 4, Float32} = [
        1 0 1 0;
        0 1 0 1;
        0 0 1 0;
        0 0 0 1;
]

const H::SMatrix{2, 4, Float32} = [
        1 0 0 0;
        0 1 0 0;
]

const Q::SMatrix{4, 4, Float32} = [
    1 0 2 0;
    0 1 0 2;
    2 0 4 0;
    0 2 0 4;
].*1.0

const R::SMatrix{2, 2, Float32} = [
    5 0;
    0 5;
]

active_ids = Int[]

function predict!(kf::KalmanFilter)
    kf.x = F*kf.x
    kf.P = F*kf.P*F' + Q
    nothing
end

function out_of_frame(x)
    return x[1] < 0 || x[1] > 1600 || x[2] < 0 || x[2] > 2560
end

# v0 = (p1 - p0)/Δt = p1-p0
# Cov(p1) = Cov(p0) = R
# Cov(v0) = 1/Δt^2 Cov(p1-p0) = 2R/Δt^2
function init_kf(detection::SVector{2, Float32})::KalmanFilter
    σ_pos = 3
    σ_v = sqrt(2)*σ_pos #Δt = 1/1frame
    return KalmanFilter(
        [detection..., 0, 5], # TODO about 5px/frame mean upwards velocity
        [
            σ_pos^2 0 0 0; # TODO improve guess for these values
            0 σ_pos^2 0 0;
            0 0 σ_v^2 0; # great uncertainty about the x-velocity
            0 0 0 σ_v^2; # some  uncertainty about the y-velocity [0, 2-6 px/frame]
        ]
    )
end

# https://en.wikipedia.org/wiki/Kalman_filter#Details
function update!(kf::KalmanFilter, measurement::SVector{2, Float32})
    y = measurement - H*kf.x # no measurement for the velocity
    S = H*kf.P*H' + R
    K = kf.P*H'*inv(S)
    kf.x = kf.x + K*y
    kf.P = (I - K*H)*kf.P
    # residual = z - H*kf.x
end

# function associate(active_ids::Vector{Int}, midpoints::Vector{SVector{2, Float32}})
#     if iszero(length(active_ids))
#         return Int[], Int[], midpoints
#     end

#     # building of cost matrix
#     # cost_matrix = zeros(length(active_ids), length(midpoints)) # TODO: make this sparse
#     cost_matrix = Matrix{Union{Missing, Float32}}(missing, length(active_ids), length(midpoints))
#     for ((track_id_index, track_id), (midpoint_index, midpoint)) in Iterators.product(enumerate(active_ids), enumerate(midpoints))
#         # mahalanobis_distance = sqrt(
#         #     transpose(midpoint - tracks[track_id].kf.x[1:2]) * inv(tracks[track_id].kf.P[1:2, 1:2]) * (midpoint - tracks[track_id].kf.x[1:2])
#         # )
#         mahalanobis_distance = norm(tracks[track_id].kf.x[1:2] - midpoint)
#         if mahalanobis_distance < GATE_DIST
#             cost_matrix[track_id_index, midpoint_index] = mahalanobis_distance
#         end
#     end

#     associations, _ = hungarian(cost_matrix)
#     @assert(length(associations) == length(active_ids)) # ?

#     unmatched_track_ids = active_ids[filter(index->iszero(associations[index]), eachindex(associations))]
#     unmatched_detections = midpoints[filter(index->!(index in associations), eachindex(midpoints))]
#     return associations, unmatched_track_ids, unmatched_detections
# end

function associate(active_ids::Vector{Int}, tracks::Dict{Int, Track}, midpoints::Vector{SVector{2, Float32}})
    prediction_points = MMatrix{2, length(active_ids), Float32}(undef)
    for (col, id) in zip(eachcol(prediction_points), active_ids)
        col .= tracks[id].kf.x[1:2]
    end

    prediction_tree = KDTree(prediction_points)
    detection_tree = KDTree(midpoints)

    # [associated midpoint for tracks[active_ids[1]] (index into midpoints), 0, associated midpoint 3, 0, ...]
    associations = zeros(Int, length(active_ids))
    # unmatched_track_ids = Int[]

    for gate_dist in [30.0, 10.0, 5.0]
        for (index, track_id) in enumerate(active_ids)
            if !iszero(associations[index])
                continue
            end
            
            prediction = tracks[track_id].kf.x[1:2]
            detection_indices = inrange(detection_tree, prediction, gate_dist)

            # if length(detection_indices) == 0
                # push!(unmatched_track_ids, track_id) # wrong if using more than one gate
            if length(detection_indices) == 1
                detection = midpoints[detection_indices[1]]
                viceversa_prediction_candidates = inrange(prediction_tree, detection, gate_dist)
                # if length(viceversa_prediction_candidates) == 1 && prediction_points[:, viceversa_prediction_candidates[1]] == tracks[track_id].kf.x[1:2]
                if length(viceversa_prediction_candidates) == 1 && active_ids[viceversa_prediction_candidates[1]] == track_id
                    # we have a match!
                    associations[index] = detection_indices[1]
                end
            end
        end # for (index, track_id) in enumerate(active_ids)
    end # for gate dist

    unmatched_track_ids = findall(iszero, associations)
    # matched_track_ids = findall(i->!iszero, associations)
    unmatched_detection_ids = findall(index->!(index in associations), eachindex(midpoints))

    return associations, unmatched_track_ids, unmatched_detection_ids
end

function run_tracking(midpoints_per_frame; nsteps=nothing)
    tracks = Dict{Int, Track}()
    active_ids = Int[]
    next_id = 1

    iter = isnothing(nsteps) ? midpoints_per_frame : Iterators.take(midpoints_per_frame, nsteps)
    for (frameind, midpoints) in ProgressBar(enumerate(iter))
        # println("analysing frame $frameind")
        # prediction
        for id in active_ids
            predict!(tracks[id].kf)
        end

        associations::Vector{Int}, unmatched_track_ids, unmatched_detection_ids = associate(active_ids, tracks, midpoints)

        # update tracks
        for (id_index, midpoint_index) in enumerate(associations)
            if iszero(midpoint_index)
                continue
            end
            id, midpoint = active_ids[id_index], midpoints[midpoint_index]

            update!(tracks[id].kf, midpoint)
            tracks[id].last_seen = frameind
            tracks[id].hits += 1
            tracks[id].misses = 0
            push!(tracks[id].history, copy(tracks[id].kf.x[1:2]))
        end

        # unmatched active tracks
        for track_id in unmatched_track_ids
            tracks[track_id].misses += 1
            if tracks[track_id].misses > MAX_MISSES || out_of_frame(tracks[track_id].kf.x)
                filter!(x -> x != track_id, active_ids)
            end
            push!(tracks[track_id].history, tracks[track_id].kf.x[1:2])
        end

        # deal with unmatched detections (spawn new tracks)
        for detection_id in unmatched_detection_ids 
            detection = midpoints[detection_id]
            tracks[next_id] = Track(
                init_kf(detection),
                frameind,
                frameind,
                1,
                0,
                [detection]
            )
            push!(active_ids, next_id)
            next_id += 1
        end
    end

    return tracks
end

function mean_point_and_distance(r1::Ray, r2::Ray)
    n1, n2 = r1.n, r2.n
    d = r2.p - r1.p

    c = cross(n1, n2)
    c2 = dot(c, c)

    # d = t1*n1 + λ*c - t2*n2 | ⋅n1
    a = dot(n1, n2)
    b1 = dot(d, n1)
    b2 = dot(d, n2)

    Δ = 1 - a*a
    t1 = (b1 - a*b2) / Δ
    t2 = (a*b1 - b2) / Δ
    λ  = dot(d, c) / c2

    return (
        (r1.p + t1*n1 + r2.p + t2*n2)/2.0,
        λ * sqrt(c2),
    )
end

function triangulate_tracks(track1::Track, track2::Track,
                            camind1::Int, camind2::Int,
                            theta)# ::Vector{SVector{3,Float32}}
    start1, start2, l1, l2 = track1.start_frame, track2.start_frame, length(track1.history), length(track2.history)
    common_timerange = intersect(start1:(start1+l1), start2:(start2+l2))
    common_range_1 = intersect(1:l1, (start2 - start1+2):(start2 + l2 - start1 + 2))
    common_range_2 = intersect(1:l2, (start1 - start2+1):(start1 + l1 - start2 + 1))
    if length(common_range_1) == 0
        return SVector{3, Float32}[], MVector{0, Float32}(), 0:0
    end

    n1, n2 = 1.0, 1.33

    points3d = SVector{3, Float32}[]
    dists = MVector{length(common_range_1), Float32}(undef)
    for (i, (point1, point2)) in enumerate(zip(track1.history[common_range_1], track2.history[common_range_2]))
        r1 = waterray_from_camera(point1..., theta, camind1, n1, n2)
        r2 = waterray_from_camera(point2..., theta, camind2, n1, n2)
        mean_point, dist = mean_point_and_distance(r1, r2)
        # if dist > DIST_GATE
        #     return nothing
        # end
        dists[i] = dist
        push!(points3d, mean_point)
    end
    return points3d, dists, common_timerange
end

function construct_KDTree(tracks, frameind)
    points = [get(pair.second.history, frameind - pair.second.start_frame + 1, SVector{2, Float32}([-20., -20.])) for pair in tracks]
    # points = something.(filter(!isnothing, points)) # filtering destroys index associativity
    return KDTree(points)
end

function associate_tracks(tracks_per_camera::Vector{Dict{Int, Track}}, theta)
    tracks1, tracks2, tracks3, tracks4 = tracks_per_camera

    DIST_GATE = 3e-3
    SEARCH_RADIUS = 10

    # (frameind, camind) -> KDTree
    KDTrees::Dict{Tuple{Int, Int}, KDTree} = Dict()
    
    for track2 in Iterators.take(tracks2, 10)
        for track4 in tracks4
            points3d, dists, common_timerange = triangulate_tracks(track2.second, track4.second, 2, 4, theta)
            if length(points3d) < 2 || median(dists) > 2e-3 # 2 mm gate
                continue
            end
            # define search frame index and 
            # search in cam 1 and 3

            important_frameinds = @view common_timerange[1:length(common_timerange)-1:end]
            important_points3d = @view points3d[1:length(points3d)-1:end]
            closest_track_indices = Int[]
            for (frameind, point3d) in zip(important_frameinds, important_points3d) # camera 3
                if !haskey(KDTrees, (frameind, 3))
                    KDTrees[(frameind, 3)] = construct_KDTree(tracks3, frameind)
                end

                projected_point = project_point_onto_image_plane(point3d, 3, theta)

                closest_track_index = knn(KDTrees[(frameind, 3)], projected_point, 1)[1][1] # only the nearest neighbor
                push!(closest_track_indices, closest_track_index)

                # if length(tracks3indices) > 0
                #     println(tracks3indices)
                #     return points3d # for plotting
                # end
            end # frameind in important_frameinds
            if allequal(closest_track_indices)
                # we have a match!
                println("from 2: $(track2.first), from 3: $(closest_track_indices[1]), from 4: $(track4.first)")
            end
        end # tracks4
    end # tracks2
end

function GLMakie.lines!(ax, track::Track; kwargs...)
    GLMakie.lines!(ax, track.history, kwargs...)
end
function GLMakie.lines!(ax, tracks::Dict{Int64, Track}; kwargs...)
    x = @views reduce(vcat, (
        vcat(reinterpret(reshape, Float32, t.second.history)[1, :], NaN32)
        for t in tracks
    ))
    y = @views reduce(vcat, (
        vcat(reinterpret(reshape, Float32, t.second.history)[2, :], NaN32)
        for t in tracks
    ))

    # lines!(ax3, x, y, alpha=0.3)
    lines!(ax, x, y, kwargs...)
end

function visualise_tracks(imagefilename, tracks)
    image = load(imagefilename)[1:1600, :]

    fig = Figure()
    ax = Makie.Axis(fig[1, 1], aspect = DataAspect(), title="Camera 3 reprojections")
    image!(ax, image)
    scatter!(ax, midpoints_per_frame[1], color=:blue)
    scatter!(ax, midpoints_per_frame[2], color=:orange)
    scatter!(ax, midpoints_per_frame[3], color=:blue, alpha=0.3)
    scatter!(ax, midpoints_per_frame[4], color=:blue, alpha=0.3)
    scatter!(ax, midpoints_per_frame[5], color=:blue, alpha=0.3)
    scatter!(ax, midpoints_per_frame[6], color=:blue, alpha=0.3)
    scatter!(ax, midpoints_per_frame[7], color=:blue, alpha=0.3)
    scatter!(ax, midpoints_per_frame[8], color=:blue, alpha=0.3)

    scatter!(ax, midpoints_per_frame[21], color=:blue, alpha=0.3)
    scatter!(ax, midpoints_per_frame[22], color=:blue, alpha=0.3)
    scatter!(ax, midpoints_per_frame[23], color=:blue, alpha=0.3)
    # plot!(ax, tracks[1].history)
    # plot!(ax, tracks[2].history)
    # plot!(ax, tracks[3].history)
    plot!(ax, tracks[4].history)

    # tracks = tracks_per_camera[3]
    lines!(ax, tracks)
    x = reduce(vcat, (
        vcat(reinterpret(reshape, Float32, t.second.history)[1, :], NaN32)
        for t in tracks
    ))
    y = reduce(vcat, (
        vcat(reinterpret(reshape, Float32, t.second.history)[2, :], NaN32)
        for t in tracks
    ))

    # lines!(ax3, x, y, alpha=0.3)
    lines!(ax, x, y, alpha=1.0, color=:blue)
    scatter!(ax, x, y, alpha=0.4, color=:black)

    # from associate
    scatter!(ax, prediction_points[:, unmatched_track_ids], color=:red)
    matched_track_ids = findall(e->!iszero(e), associations)
    scatter!(ax, prediction_points[:, matched_track_ids], color=:green)
end

function test_state()
    jsonfilename = "/home/simon/mega/masterarbeit/fullrun3_200/Camera1midpoints.json"
    imagefilename = "/home/simon/mega/masterarbeit/fullrun3_200/Camera 10000.tif"
    jsonfilename = "/home/simon/mega/masterarbeit/fullrun3_200/Camera3midpoints.json"
    imagefilename = "/home/simon/mega/masterarbeit/fullrun3_200/Camera30000.tif"

    jsonfilenames = "/home/simon/mega/masterarbeit/fullrun3_200/" .* [
        "Camera1midpoints.json",
        "Camera2midpoints.json",
        "Camera3midpoints.json",
        "Camera4midpoints.json",
    ]

    nsteps = 50
    # midpoints_per_frame = JSON.parsefile(jsonfilename, Vector{Vector{SVector{2,Float32}}})
    midpoints_per_camera_per_frame = load_midpoints(jsonfilenames)
    tracks = run_tracking(midpoints_per_camera_per_frame[3]; nsteps)

    tracks_per_camera = [
        run_tracking(
            midpoints_per_frame,
            nsteps=nsteps) for midpoints_per_frame in midpoints_per_camera_per_frame
    ]
    # filtering!
    tracks_per_camera = [
        filter!(pair->length(pair.second.history) > 4, tracks) for tracks in tracks_per_camera
    ]
    # tracks_per_camera[3] = run_tracking(midpoints_per_frame, nsteps=nsteps)

    # for id in active_ids
    #     predict!(tracks[id].kf)
    # end
    # prediction_points = MMatrix{2, length(active_ids), Float32}(undef)
    # for (col, id) in zip(eachcol(prediction_points), active_ids)
    #     col .= tracks[id].kf.x[1:2]
    # end
    # prediction_points[:, unmatched_track_ids]

    # # tracks3 : 2231, 308, 1356 reihenfolge: 308, 1356, 2231
    # itrack = Track(
    #     # next_id,
    #     init_kf(tracks_per_camera[3][308].history[1]),
    #     1,
    #     1,
    #     1,
    #     0,
    #     [tracks_per_camera[3][308].history[1]]
    # )   
    # for i in 2:7
    #     predict!(itrack.kf)
    #     update!(itrack.kf, tracks_per_camera[3][308].history[i])
    # end
    # predict!(itrack.kf)
    imagefilenames = "/home/simon/mega/masterarbeit/calib/" .* [
        "Camera 10509.tif",
        "Camera 20509.tif",
        "Camera30509.tif",
        "Camera 40509.tif",
    ]


    casepath = "/home/simon/mega/masterarbeit/fullrun3_200/"
    imagefilenames = casepath .* [
        "Camera 10000.tif",
        "Camera 20000.tif",
        "Camera30000.tif",
        "Camera 40000.tif",
    ]
    images = [load(filename)[1:1600, :] for filename in imagefilenames]

    fig = Figure()
    ax1 = Makie.Axis(fig[1, 1], aspect = DataAspect(), title="Camera 1 matched tracks from triangulations", yreversed = true)
    ax2 = Makie.Axis(fig[1, 2], aspect = DataAspect(), title="Camera 2 selected track", yreversed = true)
    ax3 = Makie.Axis(fig[2, 1], aspect = DataAspect(), title="Camera 3 matched tracks from triangulations", yreversed = true)
    ax4 = Makie.Axis(fig[2, 2], aspect = DataAspect(), title="Camera 4 matched tracks from epipolar lines", yreversed = true)
    for (ax, image) in zip([ax1, ax2, ax3, ax4], images)
        image!(ax, transpose(image), uv_transform = :flip_y)
    end

    # lines!(ax2, tracks_per_camera[2], alpha = 0.2)
    lines!(ax1, tracks_per_camera[1])
    lines!(ax2, tracks_per_camera[2])
    lines!(ax3, tracks_per_camera[3])
    lines!(ax4, tracks_per_camera[4])

    track2id = 6
    track2 = Pair(track2id, tracks2[track2id])
    lines!(ax2, track2.second)

    track4 = Pair(687, tracks4[1103])
    track4 = Pair(1291, tracks4[1291])
    track4 = Pair(1212, tracks4[1212])
    track4 = Pair(1179, tracks4[1179])
    track4 = Pair(2224, tracks_per_camera[4][2224])
    track4 = Pair(3117, tracks_per_camera[4][3117])
    track4 = Pair(603, tracks_per_camera[4][603])
    track4 = Pair(391, tracks_per_camera[4][391])
    lines!(ax4, track4.second)

    trackA = track2.second
    trackB = track4.second

    # projection to camera 3
    projected_points = project_pointcloud_onto_image_plane(points3d, 3, theta)
    lines!(ax3, projected_points, color=:red)
    close_tracks3 = filter_tracks_by_rect(tracks3, 1090, 800, 50, 400)
    close_tracks3 = filter_tracks_by_rect(tracks3, 1215, 878, 10, 4)
    lines!(ax3, close_tracks3)
    # projection to camera 1
    projected_points = project_pointcloud_onto_image_plane(points3d, 1, theta)
    lines!(ax1, projected_points, color=:red)
    close_tracks1 = filter_tracks_by_rect(tracks1, 1000, 900, 400, 100)
    lines!(ax1, close_tracks1)

    interesting_tracks = filter_tracks_by_rect(tracks3, 1070, 865, 30, 5)
    interesting_track = tracks3[2703]
    

    # project distinct onto 1 and 3
    # p2 = (100, 800)
    # p4 = (45, 750)
    p2 = (1435, 342)
    p4 = (1338, 852)
    r2 = waterray_from_camera(p2..., theta, 2, 1.0, 1.33)
    r4 = waterray_from_camera(p4..., theta, 4, 1.0, 1.33)
    mean_point, dist = mean_point_and_distance(r2, r4)
    p1 = project_point_onto_image_plane(mean_point, 1, theta)
    p3 = project_point_onto_image_plane(mean_point, 3, theta)
    scatter!(ax1, p1, color=:red)
    scatter!(ax2, p2, color=:red)
    scatter!(ax3, p3, color=:red)
    scatter!(ax4, p4, color=:red)

    # lines!(ax2, tracks2, transpose=true)
    # lines!(ax2, track2.second)

    # lines!(ax4, track4.second)

    scatter!(ax2, midpoints_per_camera_per_frame[2][1])
    scatter!(ax3, midpoints_per_camera_per_frame[3][1])

    # individual tracks
    lines!(ax2, tracks_per_camera[2][5])
    lines!(ax3, tracks_per_camera[3][2159])
    
end


function debug_3d()
    fig3d = Figure()
    ax3d = Makie.Axis3(fig3d[1, 1], aspect = :data)
    lines!([r1.p, r1.p+r1.n*1])
    lines!([r2.p, r2.p+r2.n])
    scatter!([p1, p2])
    scatter!(theta.cameraposes[:, 4:6])

    corner_points = [[0, 0], [2560, 0], [2560, 1600], [0, 1600]]
    view_window_points = [[], []]
    for p in corner_points
        rc2 = waterray_from_camera(p..., theta, 2, 1.0, 1.33)
        rc4 = waterray_from_camera(p..., theta, 4, 1.0, 1.33)

        push!(view_window_points[1], rc2.p)
        push!(view_window_points[2], rc4.p)

        lines!([rc2.p, rc2.p+rc2.n], color=:black)
        lines!([rc4.p, rc4.p+rc4.n], color=:black)
    end
    spargerray2 = waterray_from_camera(0, 800, theta, 2, 1.0, 1.33)
    spargerray4 = waterray_from_camera(0, 800, theta, 4, 1.0, 1.33)
    lines!([spargerray2.p, spargerray2.p + spargerray2.n])
    lines!([spargerray4.p, spargerray4.p + spargerray4.n])

    lines!([view_window_points[1]..., view_window_points[1][1]], color=:black)
    lines!([view_window_points[2]..., view_window_points[2][1]], color=:black)
end

function filter_tracks_by_rect(tracks, x, y, w, h)
    point_inside_rect(px, py, x, y, w, h) = px>=x && py >=y && px <=x+w && py <=y+h
    track_inside_rect(track, x, y, w, h) = any([point_inside_rect(point..., x, y, w, h) for point in track.history])

    tracks_inside = filter(pair->track_inside_rect(pair.second, 1200, 1500, 100, 200), tracks3)
    return tracks_inside
end

# from 2: 1703, from 3: 45, from 4: 2788
# from 2: 1703, from 3: 3374, from 4: 567
# from 2: 1703, from 3: 4214, from 4: 330
# from 2: 1703, from 3: 45, from 4: 2911
# from 2: 1703, from 3: 4291, from 4: 2784
