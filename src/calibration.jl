# 1. charuco edge detection & association with index
# maybe detection data structure with u,v, cameraind, frameind?

using ComponentArrays
using CondaPkg
using ImageIO
using ImageInTerminal
using Images
using LinearAlgebra
using Pkg
using PythonCall
using StaticArrays
using Optimization
using Rotations
using NonlinearSolve, ADTypes
using JSON

# (1, 2) (1, 3) (1, 4) (2, 3) (2, 4) (3, 4)
const pairings = [(i, j) for i in 1:4 for j in i+1:4]

const cv2 = pyimport("cv2")
const np = pyimport("numpy")

# datastructure that holds a ray in 3d space
struct Ray{T}
    n::SVector{3,T}
    p::SVector{3,T}
end

struct Interface{T}
    n::SVector{3,T}
    d::T # d = dot(-p,n), where p is a point on the plane. note the minus!
end

# img = Images.load("../calib/Camera30001.tif")

# Initial Parameter setup.
# whenever parameters are switched between fixed <=> free, the theta reconstruction must
# be given as well.
function initial_guess()
    R_Id = @SMatrix [
        0 0 1;
        0 -1 0;
        1 0 0
    ]
    axis = [1; 0; 0]
    R_d1 = AngleAxis(-28.31 |> deg2rad, axis...)
    R_d2 = AngleAxis(31.87 |> deg2rad, axis...)
    R_d3 = AngleAxis(56.00 |> deg2rad, axis...)
    R_d4 = AngleAxis(124.8 |> deg2rad, axis...)
    R_I1 = MRP(R_Id * R_d1)
    R_I2 = MRP(R_Id * R_d2)
    R_I3 = MRP(R_Id * R_d3)
    R_I4 = MRP(R_Id * R_d4)

    camera1pose = @SMatrix[R_I1.x R_I1.y R_I1.z -0.776121 0.385141 0]
    camera2to4poses = @SMatrix[
        R_I2.x R_I2.y R_I2.z -0.717733 -0.391225 0;
        R_I3.x R_I3.y R_I3.z -0.445777 -0.763216 0;
        R_I4.x R_I4.y R_I4.z 0.423439 -0.715978 0
    ]
    defaultcameraparameters_free = (
    # skew = 0.0,
    )

    pixel_size = 10e-6 # 10micrometers
    lens_focal_length = 100e-3
    defaultcameraparameters_fixed = (
        k1=0.0,
        k2=0.0,
        p1=0.0,
        p2=0.0,
        k3=0.0,
        fx=lens_focal_length / pixel_size,
        fy=lens_focal_length / pixel_size,
        cx=1280.0, # cx and cy are the pixel coordinates
        cy=800.0,  # of the image part, without the bottom
    )
    # cameraparameters_free = [defaultcameraparameters_free for _ in 1:4]
    cameraparameters_fixed = [defaultcameraparameters_fixed for _ in 1:4]

    interfaces = (
        interface12=(n=[-1, 0, 0], d=-0.53 / 2),
        interface34=(n=[0, -1, 0], d=-0.53 / 2)
    )
    free_parameters = ComponentArray(
        cameraposes=camera2to4poses,
        # cameraparameters=cameraparameters_free,
    )
    fixed_parameters = ComponentArray(
        cameraposes=camera1pose,
        cameraparameters=cameraparameters_fixed,
        interfaces=interfaces,
    )

    return free_parameters, fixed_parameters
end


# whenever parameters are switched between fixed <=> free, the theta reconstruction must
# be given as well.
function merge_free_and_fixed_parameters(free_parameters, fixed_parameters)
    cameraparameters = [
        (
            k1=fixed_parameters.cameraparameters[i].k1,
            k2=fixed_parameters.cameraparameters[i].k2,
            p1=fixed_parameters.cameraparameters[i].p1,
            p2=fixed_parameters.cameraparameters[i].p2,
            k3=fixed_parameters.cameraparameters[i].k3,
            fx=fixed_parameters.cameraparameters[i].fx,
            fy=fixed_parameters.cameraparameters[i].fy,
            cx=fixed_parameters.cameraparameters[i].cx,
            cy=fixed_parameters.cameraparameters[i].cy,
        )
        for i = 1:4
    ]
    theta = ComponentArray(
        cameraposes=vcat(fixed_parameters.cameraposes, free_parameters.cameraposes),
        cameraparameters=cameraparameters,
        interfaces=fixed_parameters.interfaces,
    )
    return theta
end




# 2. water ray function
# 2.1 air ray from camera
# 2.2 interface intersection
# 2.3 ray refraction

# 2.1 air ray from camera

# this function returns the undistorted points on the camera image plane from the distortion coefficients
# u, v are normalized coordinates
function undistort_point(ud, vd, cameraparameters)
    k1, k2, p1, p2, k3 = cameraparameters.k1, cameraparameters.k2, cameraparameters.p1, cameraparameters.p2, cameraparameters.k3

    u, v = ud, vd
    for _ in 1:5 # if it's enough for OpenCV...
        r2 = u^2 + v^2
        radial = 1 + k1 * r2 + k2 * r2^2 + k3 * r2^3
        du = 2 * p1 * u * v + p2 * (r2 + 2 * u^2)
        dv = p1 * (r2 + 2 * v^2) + 2 * p2 * u * v
        u = (ud - du) / radial
        v = (vd - dv) / radial
    end
    return u, v # normalized undistorted coords
end

function distort_point(u, v, cameraparameters)
    k1, k2, p1, p2, k3 = cameraparameters.k1, cameraparameters.k2, cameraparameters.p1, cameraparameters.p2, cameraparameters.k3

    r2 = u^2 + v^2
    radial = 1 + k1 * r2 + k2 * r2^2 + k3 * r2^3
    du = 2 * p1 * u * v + p2 * (r2 + 2 * u^2)
    dv = p1 * (r2 + 2 * v^2) + 2 * p2 * u * v

    return u * radial + du, v * radial + dv
end

# test distortion / undistortion
function distortionundistortiontest()
    testcameraparameters = (
        fx=55.0,
        fy=45.0,
        cx=1000,
        cy=1250,
        k1=0.1,
        k2=-0.2,
        p1=0.12,
        p2=-0.04,
        k3=0.01,
    )
    u, v = 0.05, -0.03
    ud, vd = distort_point(u, v, testcameraparameters)

    uu, vu = undistort_point(ud, vd, testcameraparameters)

    println("u: $u, v: $v")
    println("ud: $ud, vd: $vd")
    println("uu: $uu, vu: $vu")
    println("error u: $(u - uu); error v: $(v - vu)")
    println("rel. error u: $((u - uu) / u); rel. error v: $((v - vu) / v)")
end

function pixelcoords2normalisedcoords(x, y, cameraparameters)
    fx, fy, cx, cy = cameraparameters.fx, cameraparameters.fy, cameraparameters.cx, cameraparameters.cy

    u = (x - cx) / fx
    v = (y - cy) / fy
    return u, v
end

# function R3toso3(w::SVector{3})::SMatrix{3, 3}
#     return @SMatrix [
#             0 -w[3]  w[2];
#          w[3]     0 -w[1];
#         -w[2]  w[1]     0;
#     ]
# end

# function 

# returns a description in world coordinates of the ray that pierces through
# the point described by the pixelcoordinates on the cameras' image plane
function airray_from_camera(x, y, theta, cameraind)::Ray
    cameraparameters = theta.cameraparameters[cameraind]

    # convert to normalised coordinates
    u, v = pixelcoords2normalisedcoords(x, y, cameraparameters)

    # undistort (u,v)
    u, v = undistort_point(u, v, cameraparameters)

    # Rotation matrix rotating from camera frame to inertial frame
    R_IC = MRP(theta.cameraposes[cameraind, 1:3]...)

    # the norm of the normal vector
    norm = sqrt(u^2 + v^2 + 1)
    # normal vector
    n = R_IC * [u, v, 1] ./ norm
    # center of camera and point on ray
    p = theta.cameraposes[cameraind, 4:6]
    return Ray{eltype(n)}(n, p)
end

# returns the point at which the ray and the interface (plane) intersect
function intersect_ray_with_interface(ray::Ray, interface::Interface)
    denom = dot(interface.n, ray.n) # TODO check for zero breaks AD?
    lambda = -(dot(interface.n, ray.p) + interface.d) / denom
    return ray.p + lambda * ray.n
end



# the interface normal must point to the side where the light is coming from!!
function refract_ray(ray::Ray, interface::Interface, n1::Float64, n2::Float64)::Ray
    # apply Snellius' Law
    cosalpha = -dot(interface.n, ray.n)
    @assert(cosalpha > 0)

    ratio = n1 / n2 # ratio of indices of refraction
    # refracted normalized direction vector
    n_refract = ratio * ray.n + (ratio * cosalpha - sqrt(1 - ratio^2 * (1 - cosalpha^2))) * interface.n

    intersection_point = intersect_ray_with_interface(ray, interface)

    return Ray(n_refract, intersection_point)
end

function testrefraction()
    interface = Interface([0.0; 1.0; 0.0], -1.1)
    ray = Ray([1; -1; 0] ./ sqrt(2), [-2, 2, 0]')
    refractedray = refract_ray(ray, interface, 0.9, 1)

    # https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form
    println("ray:           $ray")
    println("interface:     $interface")
    println("refracted ray: $refractedray")
end

function distancesquared_ray_ray(ray1::Ray, ray2::Ray)
    normal = cross(ray1.n, ray2.n)
    return dot(ray1.p - ray2.p, normal)^2 / dot(normal, normal)
end

function distance_ray_ray(ray1::Ray, ray2::Ray)
    normal = cross(ray1.n, ray2.n)
    return dot(ray1.p - ray2.p, normal) / norm(normal)
end

function closest_points_and_distance(ray1::Ray, ray2::Ray)
    normal = cross(ray1.n, ray2.n)
    (t1, lambda, t2) = hcat(ray1.n, normal, -ray2.n)\(ray2.p - ray1.p)
    return (ray1.p + t1*ray1.n, ray2.p + t2*ray2.n, lambda * norm(normal))
end

# function closest_points

# boardimage = board.generateImage((1000, 3500))
function create_charuco_detector_and_board()
    squares_x, squares_y = 8, 22
    printing_scale_factor::Float64 = 2745e-3 / 280
    square_len = 14e-3 * printing_scale_factor
    marker_len = 10e-3 * printing_scale_factor

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_len, marker_len, aruco_dict)

    detector_params = cv2.aruco.DetectorParameters()
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

    return charuco_detector, board
end

# this function really does not work! the refraction is wayyy to great, and
# cv2.calibrateCamera returns too great focal lengths
function calibrate_naively(charuco_detector, charuco_board)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # kernel = np.ones((6,6), np.uint8)
    # eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    # closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(img)
    object_points_collected = pybuiltins.list()
    image_points_collected = pybuiltins.list()

    for filepath in ["../calib/Camera30029.tif",
        "../calib/Camera30059.tif",
        "../calib/Camera30089.tif",
        "../calib/Camera30119.tif",
        "../calib/Camera30149.tif",
        "../calib/Camera30179.tif",
        "../calib/Camera30209.tif",
        "../calib/Camera30239.tif",
        "../calib/Camera30269.tif",
        "../calib/Camera30299.tif",
        "../calib/Camera30329.tif",
        "../calib/Camera30359.tif",
        "../calib/Camera30389.tif",
        "../calib/Camera30419.tif",
        "../calib/Camera30449.tif",
        "../calib/Camera30479.tif",
    ]
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        # ids, corners = uniquely_identifiable_points_from_image_of_charuco_board(img, charuco_detector)
        corners, ids, _, _ = charuco_detector.detectBoard(img)

        if pyis(ids, pybuiltins.None)
            println("for filepath $filepath length(ids) == 0!")
            break
        end
        # ids_collected = vcat(ids_collected, ids)
        # corners_collected = vcat(corners_collected, corners)
        # ids_collected = np.hstack((ids_collected, ids))
        # corners_collected = np.vstack((corners_collected, corners))

        object_points, image_points = charuco_board.matchImagePoints(corners, ids)
        object_points_collected.append(object_points)
        image_points_collected.append(image_points)
    end

    return cv2.calibrateCamera(object_points_collected, image_points_collected, (2560, 1692), pybuiltins.None, pybuiltins.None)
end


function visualise_charuco_detection(filepath, charuco_detector)
    img = cv2.imread(filepath)
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(img)
    # visualisation
    vis = img.copy()
    cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
    cv2.aruco.drawDetectedCornersCharuco(
        vis,
        np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2),
        charuco_ids, (0, 0, 255))

    imshow(vis)
end

function imshow(img)
    cv2.imshow("image", img)
    while pyconvert(Int, cv2.waitKey(0)) != 27
        sleep(0.1)
    end
    cv2.destroyAllWindows()
end

function uniquely_identifiable_points_from_image_of_charuco_board(img, charuco_detector)::Tuple{Matrix{Float64},Vector{Int}}
    # charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(img)
    corners, ids, _, _ = charuco_detector.detectBoard(img)
    # board: 8x22
    # internal_corner_count = 21*7
    # marker_count = 4*22
    # marker_corner_count = 4*marker_count
    # total_corner_count = marker_corner_count + internal_corner_count

    # ids::Vector{Int} = [pyconvert(Int, ind) for ind in charuco_ids]

    # for markerid in marker_ids
    #     # println("$markerid")
    #     markerid = pyconvert(Int, markerid)
    #     distributed_ids = distribute_marker_corner_id(markerid)

    #     for did in distributed_ids
    #         push!(ids, did)
    #     end
    # end

    # corners::Matrix{Float64} = pyconvert(Matrix{Float64}, charuco_corners)

    # for fourmarkercorners in marker_corners
    #     fourmarkercorners = pyconvert(Matrix{Float64}, fourmarkercorners[0])

    #     corners = vcat(corners, fourmarkercorners)
    # end

    if pyis(ids, pybuiltins.None)
        return Matrix{Float64}(undef, 0, 0), Int[]

    end

    corners = pyconvert(Matrix{Float64}, corners)
    ids = pyconvert(Vector{Int}, ids)

    return corners, ids
end

function distribute_marker_corner_id(id::Int)
    # four ids per marker square,
    # 200 is greater than the corner ids max possible number
    return [1, 2, 3, 4] .+ (200 + id * 4)
end

# detect unique points on charuco board
# construct correspondences where the same point has been found by two different
# cameras
function detect_and_intersect()
    filenameexpressions = [r"Camera 1\d\d\d\d.tif", r"Camera 2\d\d\d\d.tif", r"Camera3\d\d\d\d.tif", r"Camera 4\d\d\d\d.tif"]
    directory = "../calib/"
    filenames = readdir(directory)

    # all framenumbers occurring at least in one camera
    framenumbers = filenames .|>
                   (name -> match(r"\d\d\d\d.tif", name) |>
                            match -> match.match .|>
                                     string -> parse(Int, string[1:4])
                   ) |> sort! |> unique!

    filenames = [
        filter(name -> occursin(filenameexpression, name), filenames)
        for filenameexpression in filenameexpressions
    ]

    # Map frame number -> filename for each camera
    filenames_by_frame = [
        Dict(
            parse(Int, match(r"\d{5}", name).match[2:5]) => name
            for name in camera_filenames
        )
        for camera_filenames in filenames
    ]

    charuco_detector, _ = create_charuco_detector_and_board()

    detections_list = []
    intersections_list = []
    # for concurrent_framenames in zip(filenames...)
    for framenumber in framenumbers
        # for concurrent_framenames in collect(zip(filenames...))[1:4] # TODO
        # framenames: ("Camera 10029.tif", "Camera 20029.tif", "Camera30029.tif", "Camera 40029.tif")

        detections = []
        # filenames_for_camera are equivalent to filenames_by_frame[cameraind]
        for filenames_for_camera in filenames_by_frame
            framename = get(filenames_for_camera, framenumber, nothing)

            if framename === nothing
                push!(detections, (corners=Matrix{Float64}(undef, 0, 0), ids=Int[]))
            else
                img = cv2.imread(directory * framename, cv2.IMREAD_GRAYSCALE)
                corners, ids = uniquely_identifiable_points_from_image_of_charuco_board(img, charuco_detector)

                push!(detections, (corners=corners, ids=ids))
            end
        end
        push!(detections_list, detections)
        # end

        # see if the same point has been found by multiple cameras
        # for detections in detections_list
        # holds the ids of points that appear in a camera pair
        intersections = []
        for (i, j) in pairings
            push!(intersections, intersect(detections[i].ids, detections[j].ids)::Vector{Int})
        end
        push!(intersections_list, intersections)
    end

    return detections_list, intersections_list
end

# Optimization.jl solve for system of equations

function residuals(free_parameters, p)
    theta = merge_free_and_fixed_parameters(free_parameters, p.fixed_parameters)
    T = eltype(free_parameters)

    nair = 1.0 # indices of refraction
    nwater = 1.33

    interface12 = Interface{T}(p.fixed_parameters.interfaces.interface12.n, p.fixed_parameters.interfaces.interface12.d)
    interface34 = Interface{T}(p.fixed_parameters.interfaces.interface34.n, p.fixed_parameters.interfaces.interface34.d)

    residuals = T[]
    for (detections, intersections) in zip(p.detections_list, p.intersections_list)
        for (pairingindex, intersected_ids) in enumerate(intersections)
            cameraindA, cameraindB = pairings[pairingindex]
            interfaceA = cameraindA < 3 ? interface12 : interface34
            interfaceB = cameraindB < 3 ? interface12 : interface34

            for id in intersected_ids
                detectionindexA = searchsortedfirst(
                    detections[cameraindA].ids, id
                )
                detectionindexB = searchsortedfirst(
                    detections[cameraindB].ids, id
                )

                pointA = detections[cameraindA].corners[detectionindexA, :]
                pointB = detections[cameraindB].corners[detectionindexB, :]

                rayA = airray_from_camera(pointA..., theta, cameraindA)
                rayB = airray_from_camera(pointB..., theta, cameraindB)

                refracted_rayA = refract_ray(rayA, interfaceA, nair, nwater)
                refracted_rayB = refract_ray(rayB, interfaceB, nair, nwater)

                push!(residuals, distance_ray_ray(refracted_rayA, refracted_rayB))
            end
        end
    end

    # --- ridge / prior terms on all free quantities ---
    position_priors = T[]
    for i in 1:size(free_parameters.cameraposes, 1)
        Δ = free_parameters.cameraposes[i, 4:6] .- p.prior.cameraposes[i, 4:6]
        append!(position_priors, Δ ./ p.sigma_position)
    end

    # distortion_ridge = T[]
    # for cp in free_parameters.cameraparameters
    #     for k in (cp.k1, cp.k2, cp.p1, cp.p2, cp.k3)
    #         push!(distortion_ridge, k / p.sigma_distortion)
    #     end
    # end

    # return vcat(residuals, position_priors, distortion_ridge)
    return vcat(residuals, position_priors)
end

# function main()
#     detections_list, intersections_list = detect_and_intersect()
#     free_parameters, fixed_parameters = initial_guess()

#     optf = OptimizationFunction(objective, ADTypes.AutoZygote())
#     prob = OptimizationProblem(optf, free_parameters, (fixed_parameters, detections_list, intersections_list))

#     sol = solve(prob, OptimizationLBFGSB.LBFGSB())
#     println("$sol")
# end
#

function write_blender_json(theta, detections_list, intersections_list)
    cameras = Dict{String,Any}()
    ncameras = size(theta.cameraposes, 1)
    for cameraind in 1:ncameras
        # euler = RotXYZ(transpose(Diagonal([-1, 1, -1])*MRP(theta.cameraposes[cameraind, 1:3]...)))
        pos = theta.cameraposes[cameraind, 4:6]
        Rtoblender = Diagonal([1, -1, -1])
        quat = QuatRotation(MRP(theta.cameraposes[cameraind, 1:3]...) * Rtoblender)

        cameras[string(cameraind)] = Dict(
            "position" => collect(pos),
            # "rotation_euler_rad" => [euler.theta1, euler.theta2, euler.theta3],
            "rotation_quaternion" => [quat.q.s, quat.q.v1, quat.q.v2, quat.q.v3],
        )
    end

    frames = []
    # for frameind in 1:length(detections_list)
    for frameind in 1:10
        detections = Dict{String,Any}()

        detections_object = detections_list[frameind]
        intersections_object = intersections_list[frameind]

        interesting_marker_ids = unique!(vcat(intersections_object...))

        for interesting_marker_id in interesting_marker_ids
            detections_per_marker = Dict{String,Any}()

            for cameraind in 1:ncameras
                camera_detection = detections_object[cameraind]

                if interesting_marker_id in camera_detection.ids
                    interface = cameraind < 3 ? theta.interfaces.interface12 : theta.interfaces.interface34
                    interface = Interface(SVector{3,Float64}(interface.n), interface.d)

                    detection_index::Int = searchsortedfirst(camera_detection.ids, interesting_marker_id)
                    point_on_imageplane_px = camera_detection.corners[detection_index, :]

                    # write the ray
                    airray = airray_from_camera(point_on_imageplane_px..., theta, cameraind)
                    # point_on_interface = intersect_ray_with_interface(airray, interface)
                    refracted_ray = refract_ray(airray, interface, 1.0, 1.33)
                    ray = [
                        collect(theta.cameraposes[cameraind, 4:6]),
                        collect(refracted_ray.p),
                        collect(refracted_ray.p + 0.4 * refracted_ray.n),
                    ]

                    detections_per_marker[string(cameraind)] = Dict("ray" => ray)
                end
            end
            detections[string(interesting_marker_id)] = detections_per_marker
        end

        push!(frames, Dict(
            "frame" => frameind,
            "detections" => detections,
            # "triangulations" => triangulations
        ))
    end

    data = Dict(
        "cameras" => cameras,
        "frames" => frames,
    )

    # println(data)

    # json = JSON.json(data, pretty=1)
    json = JSON.json(data)

    write("generated-files/calibration.json", json)

    return json
end

detections_list, intersections_list = detect_and_intersect()

function main(detections_list, intersections_list)
    free_parameters, fixed_parameters = initial_guess()

    p = (
        fixed_parameters=fixed_parameters,
        detections_list=detections_list,
        intersections_list=intersections_list,
        prior=free_parameters,       # reference for position priors
        sigma_position=0.05,         # 1 cm, tune to your tape-measure confidence
        sigma_distortion=0.05,       # ridge strength
    )

    resid0 = residuals(free_parameters, p)          # concrete Vector{Float64}, also a sanity check it runs
    @assert eltype(resid0) === Float64               # catches type instability early    nlfun = NonlinearFunction(residuals; resid_prototype=residuals(free_parameters, p))

    nlfun = NonlinearFunction(residuals; resid_prototype=resid0)
    prob = NonlinearLeastSquaresProblem(nlfun, free_parameters, p)
    sol = solve(prob, LevenbergMarquardt(; autodiff=AutoForwardDiff()))

    println(sol)

    thetasol = merge_free_and_fixed_parameters(sol.u, fixed_parameters)
    write_blender_json(thetasol, detections_list, intersections_list)

    return sol
end
