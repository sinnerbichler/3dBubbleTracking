# code to triangulate point cloud from ray bundles

using JSON

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
            if distance < maxdistance
                push!(points, (cpA + cpB) / 2)
            end
        end
    end
    return points
end

# reprojection of pointcloud onto image plane of camera
function project_pointcouloud_onto_image_plane(points::Vector{SVector{3,T}}, cameraind, theta) where {T<:Real}

end

# this function takes 
function project_point_onto_image_plane(point::SVector{3,T}, cameraind, theta) where {T<:Real}
    cameraposition = theta.cameraposes[cameraind, 4:6]
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

    return (px, py)
end

function reconstruct_refraction_point_from_start_end_points(p1::SVector{3,T}, p2::SVector{3,T}, interface::Interface{T}, n1, n2) where {T<:Real}
    pxc::SVector{3,T} = intersect_ray_with_interface(Ray(interface.n, p1), interface)
    pxp::SVector{3,T} = intersect_ray_with_interface(Ray(interface.n, p2), interface)

    gamma::T = norm(pxc - p1)   # distance of camera from interface plane
    kappa::T = norm(pxp - p2)            # distance of point  from interface plane
    projected_connection = pxp - pxc
    projected_distance::T = norm(projected_connection) # cameraposition -> point distance projected onto interface plane

    # similar triangles
    lambda_initial_guess = gamma*projected_distance / (gamma + kappa)

    if n1 ≈ n2
        return pxc + lambda_initial_guess * projected_connection / norm(projected_connection)
    elseif n1 < n2
        bisection_limits = (lambda_initial_guess, projected_distance)
    else # n1 > n2
        bisection_limits = (0, lambda_initial_guess)
    end
    
    f(lambda) = n1 / sqrt(gamma^2 + lambda^2) - n2 / sqrt(kappa^2 + (projected_distance - lambda)^2) * (projected_distance - lambda)

    lambda = find_zero(f, bisection_limits)

    refraction_point = pxc + lambda * projected_connection / norm(projected_connection)

    return refraction_point
end

function create_pointcloud_from_json()
    jsonpath = "/home/simon/mega/masterarbeit/fullrun3_200_midpointfiles/"
    filenames::Vector{String} = [
        "Camera1midpoints.json",
        "Camera2midpoints.json",
        "Camera3midpoints.json",
        "Camera4midpoints.json"
    ]
    filenames = [jsonpath * filename for filename in filenames]

    nair = 1.0
    nwater = 1.33
    maxdistance = 10e-3 # 10mm max distance

    _, fixed_parameters = initial_guess()
    sol = run_calibration(detections_list, intersections_list)
    thetasol = merge_free_and_fixed_parameters(sol.u, fixed_parameters)

    midpoints_per_camera = load_midpoints(filenames)
    nframes = length(midpoints_per_camera[1])

    pointclouds = Vector{Vector{SVector{3,Float32}}}(undef, nframes)
    for frameind in eachindex(midpoints_per_camera[1])
        rays_this_frame = rays_for_frame(midpoints_per_camera, frameind, thetasol, nair, nwater)
        pointclouds[frameind] = pointcloud_from_frame_rays(rays_this_frame, pairings, maxdistance)
    end
    # rays: [camera, frame, ray]
    return pointclouds
end
