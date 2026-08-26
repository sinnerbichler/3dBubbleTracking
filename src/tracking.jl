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
    1 0 0 0;
    0 1 0 0;
    0 0 5 0;
    0 0 0 5;
]

const R::SMatrix{2, 2, Float32} = [
    1 0;
    0 1;
]

active_ids = Int[]

function predict!(kf::KalmanFilter)
    kf.x = F*kf.x
    kf.P = F*kf.P*F' + Q
end

function out_of_frame(x::MVector{4, Float32})
    return x[1] < 0 || x[1] > 2560 || x[2] < 0 || x[2] > 1600
end

function init_kf(detection::SVector{2, Float32})::KalmanFilter
    return KalmanFilter(
        [detection..., 0, 5], # TODO about 5px/frame mean upwards velocity
        [
            5 0 0 0; # TODO improve guess for these values
            0 5 0 0;
            0 0 1e5 0; # great uncertainty about the x-velocity
            0 0 0 1e2; # some  uncertainty about the y-velocity [0, 2-6 px/frame]
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

function associate(active_ids::Vector{Int}, midpoints::Vector{SVector{2, Float32}})
    prediction_points = MMatrix{2, length(active_ids), Float32}(undef)
    for (col, id) in zip(eachcol(prediction_points), active_ids)
        col .= tracks[id].kf.x[1:2]
    end

    prediction_tree = KDTree(prediction_points)
    detection_tree = KDTree(midpoints)

    # [associated midpoint for tracks[active_ids[1]] (index into midpoints), 0, associated midpoint 3, 0, ...]
    associations = zeros(Int, length(active_ids))
    # unmatched_track_ids = Int[]

    for gate_dist in [15, 10, 5]
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

function run_tracking(jsonfilename)
    midpoints_per_frame = JSON.parsefile(jsonfilename, Vector{Vector{SVector{2,Float32}}})

    tracks = Dict{Int, Track}()
    active_ids = Int[]
    for (frameind, midpoints) in enumerate(midpoints_per_frame[1:20])
        println("analysing frame $frameind")
        # prediction
        for id in active_ids
            predict!(tracks[id].kf)
        end

        # associating with the hungarian algorithm
        associations::Vector{Int}, unmatched_track_ids, unmatched_detection_ids = associate(active_ids, midpoints)

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
        end

        # deal with unmatched detections (spawn new tracks)
        for detection_id in unmatched_detection_ids 
            detection = midpoints[detection_id]
            next_id = length(keys(tracks)) + 1
            tracks[next_id] = Track(
                # next_id,
                init_kf(detection),
                frameind,
                frameind,
                1,
                0,
                [detection]
            )
            push!(active_ids, next_id)
        end
    end
end

function visualise_tracks(imagefilename, tracks)
    image = load(imagefilename)[1:1600, :]

    fig = Figure()
    ax = Makie.Axis(fig[1, 1], aspect = DataAspect(), title="Camera 1 reprojections")
    image!(ax, image)
    scatter!(ax, midpoints_per_frame[1], color=:blue)
    scatter!(ax, midpoints_per_frame[2], color=:orange)
    scatter!(ax, midpoints_per_frame[3])
    scatter!(ax, midpoints_per_frame[4])
    # scatter!(ax, midpoints_per_frame[5])
    # plot!(ax, tracks[1].history)
    # plot!(ax, tracks[2].history)
    # plot!(ax, tracks[3].history)
    plot!(ax, tracks[4].history)
    # h4 = tracks[4].history
    # m4 = reinterpret(reshape, Float32, h4)
    # @views lines!(ax, m4[1, :], m4[2, :])
    for i in eachindex(tracks)
        h = tracks[i].history
        m = reinterpret(reshape, Float32, h)
        @views lines!(ax, m[1, :], m[2, :])
        # lines!(ax, tracks[i].history)
    end

    # from associate
    scatter!(ax, prediction_points[:, unmatched_track_ids], color=:red)
    matched_track_ids = findall(e->!iszero(e), associations)
    scatter!(ax, prediction_points[:, matched_track_ids], color=:green)
end

function test_state()
    jsonfilename = "/home/simon/mega/masterarbeit/fullrun3_200/Camera1midpoints.json"
    imagefilename = "/home/simon/mega/masterarbeit/fullrun3_200/Camera 10000.tif"

    for id in active_ids
        predict!(tracks[id].kf)
    end
    prediction_points = MMatrix{2, length(active_ids), Float32}(undef)
    for (col, id) in zip(eachcol(prediction_points), active_ids)
        col .= tracks[id].kf.x[1:2]
    end
    prediction_points[:, unmatched_track_ids]
end
