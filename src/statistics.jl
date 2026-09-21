# code to compute the kinematics of the tracked bubbles and how they depend on the distance to their neighbours

using StaticArrays
using LinearAlgebra
using Statistics
using NearestNeighbors
using Random

include("tracklets.jl")

# volume in which the bubbles are analysed, the lower bound in z excludes the sparger region (too dense, occluded)
const DOMAIN_LO = SVector(-0.06, -0.06, -0.08)
const DOMAIN_HI = SVector(0.06, 0.06, 0.07)

inside_domain(point) = all(DOMAIN_LO .<= point) && all(point .<= DOMAIN_HI)

# the bubbles of every frame, one exclusive selection per frame
function bubbles_per_frame(groups, midpoints_per_camera_per_frame, theta, frames)
    return Dict(f => exclusive_bubbles_at(groups, f, midpoints_per_camera_per_frame, theta) for f in frames)
end

# quadratic fit of the positions of a group in the 2*half+1 frames around f, one frame is 1 ms.
# returns position (m), velocity (m/s), acceleration (m/s^2) and the rms of the fit (m), or nothing if the group is too short.
# never fit across two groups: their positions differ by a constant that would show up as acceleration
function local_kinematics(group, f; half=5)
    k = f - first(group.frames) + 1
    (k - half >= 1 && k + half <= length(group.positions)) || return nothing
    t = collect(-half:half)
    A = hcat(ones(length(t)), t, t .^ 2)
    X = reduce(hcat, Vector.(group.positions[k-half:k+half]))'
    coefficients = A \ X
    residuals = X - A * coefficients
    return (pos=SVector{3, Float64}(coefficients[1, :]), vel=SVector{3, Float64}(coefficients[2, :]) * 1e3,
            acc=SVector{3, Float64}(coefficients[3, :]) * 2e6, rms=sqrt(sum(abs2, residuals) / (3 * (length(t) - 3))))
end

# mean radial relative acceleration (m/s^2, > 0: the two bubbles accelerate away from each other) and mean radial relative
# velocity (m/s, > 0: they move apart) over all pairs of bubbles, in bins of their distance.
# the same pair shows up in many consecutive frames and every fit spans 2*half+1 frames, so the pairs are far from independent:
# the uncertainty (standard deviation) comes from a bootstrap over blocks of `block` frames
function pair_statistics(bubbles, frames; half=5, rmax=0.03, nbins=15, block=10, nboot=2000, seed=1)
    edges = range(0.0, rmax, length=nbins + 1)
    blocks = collect(Iterators.partition(frames, block))
    n = zeros(Int, length(blocks), nbins)
    sum_a = zeros(length(blocks), nbins)
    sum_v = zeros(length(blocks), nbins)

    for (ib, block_frames) in enumerate(blocks), f in block_frames
        kinematics = [local_kinematics(group, f; half) for (group, _) in bubbles[f]]
        kinematics = [k for k in kinematics if !isnothing(k) && inside_domain(k.pos)]
        length(kinematics) < 2 && continue

        points = reduce(hcat, Vector.([k.pos for k in kinematics]))
        tree = KDTree(points)
        for i in eachindex(kinematics), j in inrange(tree, points[:, i], rmax)
            j > i || continue
            r = kinematics[j].pos - kinematics[i].pos
            distance = norm(r)
            distance > 0 || continue
            bin = min(searchsortedlast(edges, distance), nbins)
            n[ib, bin] += 1
            sum_a[ib, bin] += dot(kinematics[j].acc - kinematics[i].acc, r / distance)
            sum_v[ib, bin] += dot(kinematics[j].vel - kinematics[i].vel, r / distance)
        end
    end

    rng = MersenneTwister(seed)
    boot_a = zeros(nboot, nbins)
    boot_v = zeros(nboot, nbins)
    for k in 1:nboot
        pick = rand(rng, 1:length(blocks), length(blocks))
        counts = max.(vec(sum(n[pick, :], dims=1)), 1)
        boot_a[k, :] = vec(sum(sum_a[pick, :], dims=1)) ./ counts
        boot_v[k, :] = vec(sum(sum_v[pick, :], dims=1)) ./ counts
    end
    total = max.(vec(sum(n, dims=1)), 1)
    return (r_mm=collect((edges[1:end-1] .+ edges[2:end]) ./ 2) .* 1e3, n=vec(sum(n, dims=1)),
            acc_rel=vec(sum(sum_a, dims=1)) ./ total, acc_rel_se=vec(std(boot_a, dims=1)),
            vel_rel=vec(sum(sum_v, dims=1)) ./ total, vel_rel_se=vec(std(boot_v, dims=1)))
end

function plot_pair_statistics(statistics)
    fig = Figure(size=(1000, 900))
    ax1 = Makie.Axis(fig[1, 1], ylabel="radial relative acceleration (m/s²)", title="> 0: accelerating apart")
    errorbars!(ax1, statistics.r_mm, statistics.acc_rel, statistics.acc_rel_se)
    scatterlines!(ax1, statistics.r_mm, statistics.acc_rel)
    hlines!(ax1, [0.0], color=:gray, linestyle=:dash)
    ax2 = Makie.Axis(fig[2, 1], ylabel="radial relative velocity (m/s)", title="> 0: moving apart")
    errorbars!(ax2, statistics.r_mm, statistics.vel_rel, statistics.vel_rel_se)
    scatterlines!(ax2, statistics.r_mm, statistics.vel_rel)
    hlines!(ax2, [0.0], color=:gray, linestyle=:dash)
    ax3 = Makie.Axis(fig[3, 1], xlabel="distance of the bubble centres (mm)", ylabel="number of pairs")
    barplot!(ax3, statistics.r_mm, statistics.n)
    return fig
end
