# figures of docs/3d_tracking.md
# usage: julia --project=. docs/make_figures.jl
# needs generated-files/tracklet_state_frames1-150.jls and tracklet_results_frames1-150.jls, which are written by
# track_bubbles / pair_statistics (see docs/3d_tracking.md, section "how to run")

using GLMakie
using Serialization

include("../src/statistics.jl")

figures = joinpath(@__DIR__, "figures")
mkpath(figures)

state = deserialize(joinpath(@__DIR__, "../generated-files/tracklet_state_frames1-150.jls"))
results = deserialize(joinpath(@__DIR__, "../generated-files/tracklet_results_frames1-150.jls"))
groups, theta = state.groups, state.theta

casepath = "/home/simon/mega/masterarbeit/fullrun3_200/"
midpoints_per_camera_per_frame = load_midpoints(casepath .* ["Camera$(c)midpoints.json" for c in 1:4])

# the 4-view matches of frame 1 (match_bubbles), how well do their four rays meet for different refractive indices?
# the matches were found with n = 1.33, so this is an indication and not a proof
function figure_refraction()
    accepted = match_bubbles(midpoints_per_camera_per_frame, 1, theta; seed_gate=15.0f0, accept_gate=10.0f0, dist_gate=3e-3, min_views=4)
    indices = [1.0, 1.1, 1.2, 1.25, 1.3, 1.33, 1.36, 1.4, 1.45, 1.5]
    residuals = map(indices) do nwater
        errors = Float64[]
        for tup in accepted
            rays = [waterray_from_camera(midpoints_per_camera_per_frame[c][1][tup[c]]..., theta, c, 1.0, nwater) for c in 1:4]
            X = triangulate_rays(rays)
            for (c, ray) in enumerate(rays)
                camera = SVector{3, Float64}(theta.cameraposes[c, 4:6])
                scale = theta.cameraparameters[c].fx / (norm(ray.p - camera) + norm(X - ray.p) / nwater) # px per m
                offset = X - ray.p
                push!(errors, norm(offset - dot(offset, ray.n) * ray.n) * scale)
            end
        end
        return quantile(errors, [0.5, 0.9])
    end
    fig = Figure(size=(900, 600))
    ax = Makie.Axis(fig[1, 1], xlabel="refractive index of the water", ylabel="distance of the rays from their common point (px)",
                    title="4-view matches of frame 1", yscale=log10)
    scatterlines!(ax, indices, first.(residuals), label="median")
    scatterlines!(ax, indices, last.(residuals), label="90th percentile")
    vlines!(ax, [1.33], color=:gray, linestyle=:dash)
    axislegend(ax, position=:ct)
    save(joinpath(figures, "refraction_index.png"), fig)
end

# the systematic error is not constant along a track: reprojection error of the frame by frame 4-view triangulation
# of the best tracklet groups of one window
function figure_offset_drift()
    midpoints = [midpoints_per_camera_per_frame[c][1:30] for c in 1:4]
    tracks_per_camera = [run_tracking(midpoints[c]) for c in 1:4]
    tracks_per_camera = [filter!(p -> p.second.hits > 4 && length(p.second.history) > 4, tracks) for tracks in tracks_per_camera]
    selected = filter(g -> all(!=(0), g.ids), associate_tracklets(tracks_per_camera, theta, 1:30))
    sort!(selected, by = g -> -length(g.fit.frames))
    fig = Figure(size=(900, 600))
    ax = Makie.Axis(fig[1, 1], xlabel="frame", ylabel="reprojection error of the 4-view triangulation (px)",
                    title="the systematic error changes along a track")
    for group in selected[1:8]
        _, errors, common = triangulate_track_group([tracks_per_camera[c][group.ids[c]] for c in 1:4], [1, 2, 3, 4], theta)
        scatterlines!(ax, collect(common), Float64.(errors), markersize=5)
    end
    hlines!(ax, [ACCEPT_GATE], color=:gray, linestyle=:dash)
    save(joinpath(figures, "offset_drift.png"), fig)
end

function figure_yield()
    yield = results.yield
    fig = Figure(size=(900, 600))
    ax = Makie.Axis(fig[1, 1], xlabel="frame", ylabel="number per frame", title="tracked bubbles per frame")
    lines!(ax, first.(yield), getindex.(yield, 2), label="bubbles")
    lines!(ax, first.(yield), getindex.(yield, 3), label="with velocity (5 frames)")
    lines!(ax, first.(yield), getindex.(yield, 4), label="with acceleration (11 frames)")
    ylims!(ax, 0, nothing)
    axislegend(ax, position=:rb)
    save(joinpath(figures, "yield.png"), fig)
end

# the 3d tracks of the bubbles of frame f0 up to frame f1 reprojected into the four cameras. no image exists for these frames,
# the raw detections are the background: blue = f0, orange = f1. a good track starts on a blue and ends on an orange point
# red = 4-camera group, magenta = 3-camera group (drawn in the cameras that are part of the group)
function figure_tracks_cameras(f0=75, f1=85)
    bubbles = exclusive_bubbles_at(groups, f0, midpoints_per_camera_per_frame, theta)
    fig = Figure(size=(1800, 1150))
    for c in 1:4
        ax = Makie.Axis(fig[(c - 1) ÷ 2 + 1, (c - 1) % 2 + 1], aspect=DataAspect(), yreversed=true, limits=(0, 2560, 0, 1600),
                        title="camera $c, tracks of frame $f0 to $f1")
        scatter!(ax, midpoints_per_camera_per_frame[c][f0], color=:steelblue, markersize=6)
        scatter!(ax, midpoints_per_camera_per_frame[c][f1], color=:orange, markersize=6)
        for (group, _) in bubbles
            ci = findfirst(==(c), group.fit.cams)
            frames = [f for f in f0:f1 if f in group.frames]
            (isnothing(ci) || length(frames) < 2) && continue
            points = [project_point_onto_image_plane(SVector{3, Float64}(group.positions[f - first(group.frames) + 1] + group.fit.offsets[ci]), c, theta) for f in frames]
            lines!(ax, points, color=group.ncams == 4 ? :red : :magenta, linewidth=2)
        end
    end
    save(joinpath(figures, "tracks_cameras.png"), fig)
end

# the same tracks in 3d (mm), the tracks with at least 6 frames
function figure_tracks_3d(f0=75, f1=85)
    bubbles = exclusive_bubbles_at(groups, f0, midpoints_per_camera_per_frame, theta)
    fig = Figure(size=(900, 900))
    ax = Axis3(fig[1, 1], aspect=:data, xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)", title="bubble tracks, frames $f0 to $f1")
    for (group, _) in bubbles
        frames = [f for f in f0:f1 if f in group.frames]
        length(frames) >= 6 || continue
        lines!(ax, [Point3f(group.positions[f - first(group.frames) + 1] * 1e3) for f in frames], color=group.ncams == 4 ? :red : :magenta, linewidth=2)
    end
    save(joinpath(figures, "tracks_3d.png"), fig)
end

function figure_pair_statistics()
    s, s4 = results.statistics, results.statistics4
    fig = Figure(size=(900, 1100))
    ax1 = Makie.Axis(fig[1, 1], ylabel="radial relative acceleration (m/s²)", title="> 0: accelerating apart")
    errorbars!(ax1, s.r_mm, s.acc_rel, s.acc_rel_se, color=:steelblue)
    scatterlines!(ax1, s.r_mm, s.acc_rel, color=:steelblue, label="all groups")
    errorbars!(ax1, s4.r_mm, s4.acc_rel, s4.acc_rel_se, color=:orange)
    scatterlines!(ax1, s4.r_mm, s4.acc_rel, color=:orange, marker=:rect, label="4-camera groups only")
    hlines!(ax1, [0.0], color=:gray, linestyle=:dash)
    axislegend(ax1, position=:rt)
    ax2 = Makie.Axis(fig[2, 1], ylabel="radial relative velocity (mm/s)", title="> 0: moving apart")
    errorbars!(ax2, s.r_mm, s.vel_rel .* 1e3, s.vel_rel_se .* 1e3, color=:steelblue)
    scatterlines!(ax2, s.r_mm, s.vel_rel .* 1e3, color=:steelblue)
    errorbars!(ax2, s4.r_mm, s4.vel_rel .* 1e3, s4.vel_rel_se .* 1e3, color=:orange)
    scatterlines!(ax2, s4.r_mm, s4.vel_rel .* 1e3, color=:orange, marker=:rect)
    hlines!(ax2, [0.0], color=:gray, linestyle=:dash)
    ax3 = Makie.Axis(fig[3, 1], xlabel="distance of the bubble centres (mm)", ylabel="number of pairs")
    barplot!(ax3, s.r_mm, s.n, color=:steelblue)
    barplot!(ax3, s4.r_mm, s4.n, color=:orange)
    save(joinpath(figures, "pair_statistics.png"), fig)
end

figure_refraction()
figure_offset_drift()
figure_yield()
figure_tracks_cameras()
figure_tracks_3d()
figure_pair_statistics()
