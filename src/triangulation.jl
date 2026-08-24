# code to triangulate point cloud from ray bundles

using JSON
using Roots
using GLMakie
using ImageIO
using FileIO

include("calibration.jl")

function load_midpoints(filenames::Vector{String})
    return [JSON.parsefile(f, Vector{Vector{SVector{2,Float32}}}) for f in filenames]
end

function rays_for_frame(midpoints_per_camera, frame::Int, theta, n1, n2)
    return [
        [waterray_from_camera(pt..., theta, cameraind, n1, n2) for pt in midpoints_per_camera[cameraind][frame]]
        for cameraind in eachindex(midpoints_per_camera)
    ]
end

function pointcloud_from_frame_rays(rays_this_frame::Vector{Vector{Ray{T}}}, pairings, maxdistance::T) where {T<:Real}
    points = SVector{3,T}[]
    for (i, j) in pairings
        raysA, raysB = rays_this_frame[i], rays_this_frame[j]
        for (rayA, rayB) in Iterators.product(raysA, raysB)
            cpA, cpB, distance = closest_points_and_distance(rayA, rayB)
            if abs(distance) < maxdistance
                push!(points, (cpA + cpB) / 2)
            end
        end
    end
    return points
end

# reprojection of pointcloud onto image plane of camera
function project_pointcloud_onto_image_plane(points::Vector{SVector{3,T}}, cameraind, theta) where {T<:Real}
    return [project_point_onto_image_plane(point, cameraind, theta) for point in points]
end

# this function takes 
function project_point_onto_image_plane(point::SVector{3,T}, cameraind, theta) where {T<:Real}
    cameraposition = SVector{3, T}(theta.cameraposes[cameraind, 4:6])
    cameraparameters = theta.cameraparameters[cameraind]

    interface12 = Interface{T}(theta.interfaces.interface12.n, theta.interfaces.interface12.d)
    interface34 = Interface{T}(theta.interfaces.interface34.n, theta.interfaces.interface34.d)

    interface = cameraind < 3 ? interface12 : interface34

    nair::T = 1.0
    nwater::T = 1.33

    refraction_point = reconstruct_refraction_point_from_start_end_points(cameraposition, point, interface, nair, nwater)

    raydirection = refraction_point - cameraposition

    # Rotation matrix rotating from camera frame to inertial frame
    R_IC = MRP(theta.cameraposes[cameraind, 1:3]...)

    n_in_cameraframe = R_IC' * raydirection

    homogenous_coordinates = n_in_cameraframe[1:2] ./ n_in_cameraframe[3] # z=1; u, v
    distorted = distort_point(homogenous_coordinates..., cameraparameters) # distorted u, v

    (px, py) = normalisedcoords2pixelcoords(distorted..., cameraparameters)

    return @SVector[px, py]
end

function reconstruct_refraction_point_from_start_end_points(p1::SVector{3,T}, p2::SVector{3,T}, interface::Interface{T}, n1, n2) where {T<:Real}
    pxc::SVector{3,T} = intersect_ray_with_interface(Ray(interface.n, p1), interface)
    pxp::SVector{3,T} = intersect_ray_with_interface(Ray(interface.n, p2), interface)

    dist1::T = norm(pxc - p1)   # distance of camera from interface plane
    dist2::T = norm(pxp - p2)   # distance of point  from interface plane
    projected_connection = pxp - pxc # onto the interface plane
    projected_distance::T = norm(projected_connection) # cameraposition -> point distance projected onto interface plane

    # similar triangles
    lambda_initial_guess = dist1*projected_distance / (dist1 + dist2)

    if norm(cross(interface.n, (p1-p2)/norm(p1-p2))) < 1e-5 # p1->p2 || n
        return pxc
    elseif n1 ≈ n2
        return pxc + lambda_initial_guess * projected_connection / norm(projected_connection)
    elseif n1 < n2
        bisection_limits = (lambda_initial_guess, projected_distance)
    else # n1 > n2
        bisection_limits = (0, lambda_initial_guess)
    end
    
    # f is the derivative of the weighted distance function w.r.t. lambda (see thesis)
    f(lambda) = n1 / sqrt(dist1^2 + lambda^2)*lambda - n2 / sqrt(dist2^2 + (projected_distance - lambda)^2) * (projected_distance - lambda)

    lambda = find_zero(f, bisection_limits)

    refraction_point = pxc + lambda * projected_connection / norm(projected_connection)

    return refraction_point
end

function ray_casting_and_reprojecting_tests()
    # rot = MRP(RotXYZ(0, 0, pi/4))
    rot = MRP([
                  -1/sqrt(2)  0.0 1/sqrt(2);
                  1/sqrt(2) 0.0 1/sqrt(2);
                  0.0 1.0 0.0
              ])
    interface = Interface(@SVector[-1.0, 0.0, 0.0], 0.0)
    theta = ComponentArray(
        cameraposes = [rot.x rot.y rot.z -1.0 -1.0 0.0],
        cameraparameters = [(k1 = 0.0, k2 = 0.0, p1 = 0.0, p2 = 0.0, k3 = 0.0, fx = 10000.0, fy = 10000, cx = 1280.0, cy = 800.0)],
        interfaces = (interface12 = (n = interface.n, d = interface.d),interface34 = (n = interface.n, d = interface.d)),
    )
    cameraind = 1

    # first test: try with known points and verify with snells' law
    camerapos = SVector{3, Float64}(theta.cameraposes[1, 4:6])
    point = @SVector([1.0, 1.0, 0.0])
    p1 = camerapos
    p2 = point
    n1 = 1.0
    n2 = 1.33

    refraction_point = reconstruct_refraction_point_from_start_end_points(camerapos, point, interface, n1, n2)

    sinalpha = (refraction_point-camerapos)[2] / norm(refraction_point-camerapos)
    sinbeta = (point - refraction_point)[2] / norm(point - refraction_point)

    @assert(sinalpha/sinbeta ≈ n2/n1)

    # second test: cast ray from camera and reconstruct 
    pixelpoint = [1280.0 + 10.0, 800.0 + 10.0] # close to the principal point
    waterray = waterray_from_camera(pixelpoint..., theta, 1, n1, n2)
    point_along_waterray = waterray.p + 1.5*waterray.n

    refraction_point = reconstruct_refraction_point_from_start_end_points(camerapos, point_along_waterray, interface, n1, n2)

    @assert(refraction_point ≈ waterray.p)

    # third test: project point along waterray onto image plane
    on_image_plane = project_point_onto_image_plane(point_along_waterray, cameraind, theta)
    @assert(norm(collect(on_image_plane) - pixelpoint) < 1e-5)
end

function create_pointclouds_from_json()
    jsonpath = "/home/simon/mega/masterarbeit/fullrun3_200/"
    filenames::Vector{String} = [
        "Camera1midpoints.json",
        "Camera2midpoints.json",
        "Camera3midpoints.json",
        "Camera4midpoints.json"
    ]
    filenames = [jsonpath * filename for filename in filenames]

    nair = 1.0
    nwater = 1.33
    maxdistance = 1e-3

    _, fixed_parameters = initial_guess()
    sol = run_calibration(detections_list, intersections_list)
    thetasol = merge_free_and_fixed_parameters(sol.u, fixed_parameters)

    midpoints_per_camera = load_midpoints(filenames)
    nframes = length(midpoints_per_camera[1])

    pointclouds = Vector{Vector{SVector{3,Float32}}}(undef, nframes)
    for frameind in eachindex(midpoints_per_camera[1])
        rays_this_frame = rays_for_frame(midpoints_per_camera,
            frameind,
            thetasol,
            nair,
            nwater)
        pointclouds[frameind] = pointcloud_from_frame_rays(rays_this_frame,
            pairings,
            maxdistance)
    end
    # rays: [camera, frame, ray]
    return pointclouds
end

function plot_midpoints_over_frames()
    casepath = "/home/simon/mega/masterarbeit/fullrun3_200/"
    jsonfilenames::Vector{String} = [
        "Camera1midpoints.json",
        "Camera2midpoints.json",
        "Camera3midpoints.json",
        "Camera4midpoints.json"
    ]
    jsonfilenames = [casepath * filename for filename in jsonfilenames]
    midpoints_per_camera = load_midpoints(jsonfilenames)
    
    imagefilenames = casepath .* [
        "Camera 10000.tif",
        "Camera 20000.tif",
        "Camera30000.tif",
        "Camera 40000.tif",
    ]
    images = [load(filename)[1:1600, :] for filename in imagefilenames]

    fig = Figure()
    ax1 = Makie.Axis(fig[1, 1], aspect = DataAspect(), title="Camera 1 image detections")
    ax2 = Makie.Axis(fig[1, 2], aspect = DataAspect(), title="Camera 2 image detections")
    ax3 = Makie.Axis(fig[2, 1], aspect = DataAspect(), title="Camera 3 image detections")
    ax4 = Makie.Axis(fig[2, 2], aspect = DataAspect(), title="Camera 4 image detections")

    for (ax, image) in zip([ax1, ax2, ax3, ax4], images)
        image!(ax, image)
    end
    for (ax, points) in zip([ax1, ax2, ax3, ax4], midpoints_per_camera)
        scatter!(ax, points[1], markersize = 4, color=:red)
    end
end

function plot_reprojections_over_frames(pointcloud::Vector{SVector{3, Float32}}, thetasol)
    theta32 = Float32.(thetasol)
    casepath = "/home/simon/mega/masterarbeit/fullrun3_200/"
    jsonfilenames::Vector{String} = [
        "Camera1midpoints.json",
        "Camera2midpoints.json",
        "Camera3midpoints.json",
        "Camera4midpoints.json"
    ]
    jsonfilenames = [casepath * filename for filename in jsonfilenames]
    midpoints_per_camera = load_midpoints(jsonfilenames)
    
    imagefilenames = casepath .* [
        "Camera 10000.tif",
        "Camera 20000.tif",
        "Camera30000.tif",
        "Camera 40000.tif",
    ]
    images = [load(filename)[1:1600, :] for filename in imagefilenames]

    projected_points_per_camera = [project_pointcloud_onto_image_plane(pointcloud, i, theta32) for i in 1:4]

    fig = Figure()
    ax1 = Makie.Axis(fig[1, 1], aspect = DataAspect(), title="Camera 1 reprojections")
    ax2 = Makie.Axis(fig[1, 2], aspect = DataAspect(), title="Camera 2 reprojections")
    ax3 = Makie.Axis(fig[2, 1], aspect = DataAspect(), title="Camera 3 reprojections")
    ax4 = Makie.Axis(fig[2, 2], aspect = DataAspect(), title="Camera 4 reprojections")

    for (ax, image) in zip([ax1, ax2, ax3, ax4], images)
        image!(ax, image)
    end
    for (ax, points) in zip([ax1, ax2, ax3, ax4], projected_points_per_camera)
        scatter!(ax, points, markersize = 4, color=:red)
    end
end

function visualise_point_cloud(pointcloud, thetasol)
    pcfig = Figure()
    ax = GLMakie.Axis3(pcfig[1, 1], aspect=:data)
    scatter!(ax, pointcloud, markersize=1)
    scatter!(ax, thetasol.cameraposes[:, 4:6], markersize=50)
end
    
