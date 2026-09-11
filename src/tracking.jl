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

include("calibration.jl")

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
    return x[1] < 0 || x[1] > 2560 || x[2] < 0 || x[2] > 1600
end

# v0 = (p1 - p0)/Δt = p1-p0
# Cov(p1) = Cov(p0) = R
# Cov(v0) = 1/Δt^2 Cov(p1-p0) = 2R/Δt^2
function init_kf(detection::SVector{2, Float32})::KalmanFilter
    σ_pos = 1
    σ_v = sqrt(2)*σ_pos #Δt = 1/1frame
    return KalmanFilter(
        [detection..., 5, 0], # TODO about 5px/frame mean upwards velocity
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

    unmatched_track_ids = active_ids[findall(iszero, associations)]
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

function triangulate_tracks(trackA::Track, trackB::Track,
                            camind1::Int, camind2::Int,
                            theta)# ::Vector{SVector{3,Float32}}
    start1, start2, l1, l2 = trackA.start_frame, trackB.start_frame, length(trackA.history), length(trackB.history)
    common_timerange = intersect(start1:(start1+l1-1), start2:(start2+l2-1))
    common_range_1 = intersect(1:l1, (start2 - start1 + 1):(start2 + l2 - start1))
    common_range_2 = intersect(1:l2, (start1 - start2 + 1):(start1 + l1 - start2))
    if length(common_range_1) == 0
        return SVector{3, Float32}[], MVector{0, Float32}(), 0:0
    end

    n1, n2 = 1.0, 1.33

    points3d = SVector{3, Float32}[]
    dists = MVector{length(common_range_1), Float32}(undef)
    for (i, (point1, point2)) in enumerate(zip(trackA.history[common_range_1], trackB.history[common_range_2]))
        r1 = waterray_from_camera(point1..., theta, camind1, n1, n2)
        r2 = waterray_from_camera(point2..., theta, camind2, n1, n2)
        mean_point, dist = mean_point_and_distance(r1, r2)
        # p1, p2, dist = closest_points_and_distance(r1, r2)
        # mean_point = (p1+p2)/2
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
    KDTrees = Dict{Tuple{Int, Int}, KDTree}()
    
    for track2 in Iterators.take(tracks2, 10) # TODO remove
        for track4 in tracks4
            points3d, dists, common_timerange = triangulate_tracks(track2.second, track4.second, 2, 4, theta)
            if length(points3d) < 2 || median(abs.(dists)) > 5e-3 # 1 mm gate
                continue
            end
            println("candidate in tracks4 found: $(track4.first)")
            points1 = project_pointcloud_onto_image_plane(points3d, 1, theta)
            points3 = project_pointcloud_onto_image_plane(points3d, 3, theta)
            lines!(ax1, points1, color=:blue, alpha=0.5)
            lines!(ax3, points3, color=:blue, alpha=0.5)
            # define search frame index and 
            # search in cam 1 and 3

            # take the first and last points as indicator points
            important_frameinds = @view common_timerange[1:length(common_timerange)-1:end]
            important_points3d = @view points3d[1:length(points3d)-1:end]

            closest_track_indices = Int[]
            for (frameind, point3d) in zip(important_frameinds, important_points3d) # camera 3
                if !haskey(KDTrees, (frameind, 3))
                    KDTrees[(frameind, 3)] = construct_KDTree(tracks3, frameind)
                end

                projected_point = project_point_onto_image_plane(point3d, 3, theta)
                scatter!(ax3, projected_point, color=:red)

                closest_track_indices, distances = knn(KDTrees[(frameind, 3)], projected_point, 1) # only the nearest neighbor
                closest_track_index = closest_track_indices[1]
                if any(distances .> 10)
                    push!(closest_track_indices, -1) # ruin the allequal check
                    break
                end
                push!(closest_track_indices, closest_track_index)

                # if length(tracks3indices) > 0
                #     println(tracks3indices)
                #     return points3d # for plotting
                # end
            end # frameind in important_frameinds
            if allequal(closest_track_indices)
                # we have a match!
                println("from 2: $(track2.first), from 3: $(closest_track_indices[1]), from 4: $(track4.first)")
                # lines!(ax4, track4.second)
                # lines!(ax3, tracks3[closest_track_indices[1]])
                break
            end
        end # tracks4
    end # tracks2
end

function GLMakie.lines!(ax, track::Track; kwargs...)
    GLMakie.lines!(ax, track.history, kwargs...)
end
function GLMakie.lines!(ax, tracks::Dict{Int64, Track}; transpose=false, kwargs...)
    x = @views reduce(vcat, (
        vcat(reinterpret(reshape, Float32, t.second.history)[1, :], NaN32)
        for t in tracks
    ))
    y = @views reduce(vcat, (
        vcat(reinterpret(reshape, Float32, t.second.history)[2, :], NaN32)
        for t in tracks
    ))

    # lines!(ax3, x, y, alpha=0.3)
    transpose ? lines!(ax, y, x, kwargs...) : lines!(ax, x, y, kwargs...)
end

function visualise_tracks(imagefilename, tracks)
    image = transpose(load(imagefilename)[1:1600, :])

    fig = Figure()
    ax = Makie.Axis(fig[1, 1], aspect = DataAspect(), title="Camera 3 reprojections")
    image!(ax, image)
    scatter!(ax, midpoints_per_frame[1], color=:blue)
    scatter!(ax, midpoints_per_frame[2], color=:orange)
    scatter!(ax, midpoints_per_frame[3], alpha=0.2, color=:black)
    scatter!(ax, midpoints_per_frame[4], alpha=0.2, color=:black)
    scatter!(ax, midpoints_per_frame[5], alpha=0.2, color=:black)
    scatter!(ax, midpoints_per_frame[6], alpha=0.2, color=:black)
    scatter!(ax, midpoints_per_frame[7], alpha=0.2, color=:black)
    scatter!(ax, midpoints_per_frame[8], alpha=0.2, color=:black)
    # plot!(ax, tracks[1].history)
    # plot!(ax, tracks[2].history)
    # plot!(ax, tracks[3].history)
    plot!(ax, tracks[4].history)

    tracks = tracks_per_camera[3]
    x = reduce(vcat, (
        vcat(reinterpret(reshape, Float32, t.second.history)[1, :], NaN32)
        for t in tracks
    ))
    y = reduce(vcat, (
        vcat(reinterpret(reshape, Float32, t.second.history)[2, :], NaN32)
        for t in tracks
    ))

    # lines!(ax3, x, y, alpha=0.3)
    lines!(ax, x, y, alpha=0.3)

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
    midpoints_per_camera_per_frame = load_midpoints(jsonfilenames)

    tracks_per_camera = [
        run_tracking(
            midpoints_per_frame,
            nsteps=nsteps) for midpoints_per_frame in midpoints_per_camera_per_frame
    ]
    # filtering!
    tracks_per_camera = [
        filter!(pair->pair.second.hits > 4, tracks) for tracks in tracks_per_camera
    ]
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


    casepath = "/home/simon/mega/masterarbeit/fullrun3_200/"
    imagefilenames = casepath .* [
        "Camera 10000.tif",
        "Camera 20000.tif",
        "Camera30000.tif",
        "Camera 40000.tif",
    ]
    images = [load(filename)[1:1600, :] for filename in imagefilenames]

    fig = Figure()
    ax1 = Makie.Axis(fig[1, 1], aspect = DataAspect(), title="Camera 1 matched tracks from triangulations")
    ax2 = Makie.Axis(fig[1, 2], aspect = DataAspect(), title="Camera 2 selected track")
    ax3 = Makie.Axis(fig[2, 1], aspect = DataAspect(), title="Camera 3 matched tracks from triangulations")
    ax4 = Makie.Axis(fig[2, 2], aspect = DataAspect(), title="Camera 4 matched tracks from epipolar lines")
    for (ax, image) in zip([ax1, ax2, ax3, ax4], images)
        image!(ax, image)
    end

    # lines!(ax2, tracks_per_camera[2], alpha = 0.2)
    track2 = tracks_per_camera[2][3]
    lines!(ax2, track2)

    scatter!(ax2, midpoints_per_frame[1])

    # individual tracks
    lines!(ax2, tracks_per_camera[2][4])
    
end

function filter_tracks_by_rect(tracks, x, y, w, h; take_any=true)
    point_inside_rect(px, py, x, y, w, h) = px>=x && py >=y && px <=x+w && py <=y+h
    track_inside_rect(track, x, y, w, h) = take_any ? any([point_inside_rect(point..., x, y, w, h) for point in track.history]) : any([point_inside_rect(point..., x, y, w, h) for point in track.history])

    tracks_inside = filter(pair->track_inside_rect(pair.second, x, y, w, h), tracks)
    return tracks_inside
end

# from 2: 1703, from 3: 45, from 4: 2788
# from 2: 1703, from 3: 3374, from 4: 567
# from 2: 1703, from 3: 4214, from 4: 330
# from 2: 1703, from 3: 45, from 4: 2911
# from 2: 1703, from 3: 4291, from 4: 2784
#

function triangluationtest()
    tracks1, tracks2, tracks3, tracks4 = tracks_per_camera

    track2id = 9
    track2 = Pair(track2id, tracks2[track2id])
    lines!(ax2, track2.second)

    # epipolar lines
    lengths = 0.05:0.025:0.2
    lines!(ax1, epipolar_curve(track2.second.history[1]..., lengths, 2, 1, theta))
    lines!(ax3, epipolar_curve(track2.second.history[1]..., lengths, 2, 3, theta))
    lines!(ax4, epipolar_curve(track2.second.history[1]..., lengths, 2, 4, theta))

    points1 = midpoints_per_camera_per_frame[1][1]
    points3 = midpoints_per_camera_per_frame[3][1]
    points4 = midpoints_per_camera_per_frame[4][1]
    scatter!(ax1, points1, color=:green, alpha=0.4)
    scatter!(ax3, points3, color=:green, alpha=0.4)
    scatter!(ax4, points4, color=:green, alpha=0.4)

    tree1 = KDTree(points1)
    tree3 = KDTree(points3)

    ray2 = waterray_from_camera(track2.second.history[1]..., theta, 2, 1.0, 1.33)
    close_points_4 = []
    for midpoint in points4
        ray4 = waterray_from_camera(midpoint..., theta, 4, 1.0, 1.33)
        # p2, _, distance = closest_points_and_distance(ray2, ray4) # or mean_point_and_distance?
        p2, distance = mean_point_and_distance(ray2, ray4)

        if abs(distance) >= 1e-3
            continue
        end
        push!(close_points_4, midpoint)

        # reproject onto 1 and 3
        p1 = project_point_onto_image_plane(p2, 1, theta)
        p3 = project_point_onto_image_plane(p2, 3, theta)

        scatter!(ax1, p1, color=:blue)
        scatter!(ax3, p3, color=:blue)
    end
    scatter!(ax4, close_points_4, color=:red)
    # scatter!(ax4, close_points_4, color=:black)

    # for midpoint in close_points_4
    #     ray4 = waterray_from_camera(midpoint..., theta, 4, 1.0, 1.33)

    # end

    # triangulation by hand
    p1 = @SVector[397, 880]
    p2 = track2.second.history[1]
    # p2 = @SVector[397, 880]
    p3 = @SVector[460, 835] # jackpot!!!!
    p4 = @SVector[2427, 216]

    r1 = waterray_from_camera(p1..., theta, 1, 1.0, 1.33)
    r2 = waterray_from_camera(p2..., theta, 2, 1.0, 1.33)
    r3 = waterray_from_camera(p3..., theta, 3, 1.0, 1.33)
    r4 = waterray_from_camera(p4..., theta, 4, 1.0, 1.33)
    pd, d = mean_point_and_distance(r1, r3)

    p1 = project_point_onto_image_plane(pd, 1, theta)
    p2 = project_point_onto_image_plane(pd, 2, theta)
    p3 = project_point_onto_image_plane(pd, 3, theta)
    p4 = project_point_onto_image_plane(pd, 4, theta)
    
    scatter!(ax1, p1, color=:yellow)
    scatter!(ax2, p2, color=:yellow)
    scatter!(ax3, p3, color=:yellow)
    scatter!(ax4, p4, color=:yellow)
end
