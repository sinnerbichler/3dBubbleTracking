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

const cv2 = pyimport("cv2")
const np = pyimport("numpy")

# img = Images.load("../calib/Camera30001.tif")

# Initial Parameter setup.
# whenever parameters are switched between fixed <=> free, the theta reconstruction must
# be given as well.
function initial_guess()
    R_Id = @SMatrix [
        0 0 1;
        0 -1 0;
        1 0 0;
    ]
    axis = [1; 0; 0];
    R_d1 = AngleAxis(-28.31 |> deg2rad, axis...)
    R_d2 = AngleAxis( 31.87 |> deg2rad, axis...)
    R_d3 = AngleAxis( 56.00 |> deg2rad, axis...)
    R_d4 = AngleAxis( 124.8 |> deg2rad, axis...)
    R_I1 = MRP(R_Id * R_d1)
    R_I2 = MRP(R_Id * R_d2)
    R_I3 = MRP(R_Id * R_d3)
    R_I4 = MRP(R_Id * R_d4)

    camera1pose = [R_I1.x R_I1.y R_I1.z -0.776121  0.385141 0];
    camera2to4poses = @SMatrix[
        R_I2.x R_I2.y R_I2.z -0.717733 -0.391225 0;
        R_I3.x R_I3.y R_I3.z -0.445777 -0.763216 0;
        R_I4.x R_I4.y R_I4.z  0.423439 -0.715978 0;
    ]
    defaultcameraparameters = (
        fx=100.0,
        fy=100.0,
        cx=1280.0, # cx and cy are the pixel coordinates
        cy=800.0,  # of the image part, without the bottom
                            # info row
        # skew = 0.0,

        k1=0.0,
        k2=0.0,
        p1=0.0,
        p2=0.0,
        k3=0.0,
    )
    cameraparameters = [defaultcameraparameters for _ in 1:4]

    interfaces = (
        interface12=(n=[-1, 0, 0], p=-0.53/2),
        interface34=(n=[0, -1, 0], p=-0.53/2)
    )
    free_parameters = ComponentArray(
        cameraposes=camera2to4poses,
        cameraparameters=cameraparameters,
        interfaces=interfaces,
    )
    fixed_parameters = ComponentArray(
        camera1pose = camera1pose
    )

    return free_parameters, fixed_parameters
end


# whenever parameters are switched between fixed <=> free, the theta reconstruction must
# be given as well.
function merge_free_and_fixed_parameters(free_parameters, fixed_parameters)
    theta = ComponentArray(
        cameraposes = vcat(fixed_parameters.camera1pose, free_parameters.cameraposes),
        cameraparameters = free_parameters.cameraparameters,
        interfaces=free_parameters.interfaces
    )
    return theta
end
    


# datastructure that holds a ray in 3d space
struct Ray
    n::SVector{3,Float64}
    p::SVector{3,Float64}
end

struct Interface
    n::SVector{3,Float64}
    d::Float64 # d = dot(-p,n), where p is a point on the plane. note the minus!
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
    R_IC = MRP(theta.cameraposes[cameraind, 1:3])

    # the norm of the normal vector
    norm = sqrt(u^2 + v^2 + 1)
    # normal vector
    n = R_IC*[u, v, 1]./norm
    # center of camera and point on ray
    p = theta.cameraposes[cameraind, 4:6]
    return Ray(n, p)
end

# returns the point at which the ray and the interface (plane) intersect
function intersect_ray_with_interface(ray::Ray, interface::Interface)::SVector{3, Float64}
    denom = dot(interface.n, ray.n) # TODO check for zero breaks AD?
    lambda::Float64 = -(dot(interface.n, ray.p) + interface.d) / denom
    return ray.p + lambda*ray.n
end

# the interface normal must point to the side where the light is coming from!!
function refract_ray(ray, interface, n1, n2)
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

function distancesquared(ray1::Ray, ray2::Ray)::Float64
    normal = cross(ray1.n, ray2.n)
    return dot(ray1.p - ray2.p, normal)^2 / (dot(normal, normal))
end

# boardimage = board.generateImage((1000, 3500))
function create_charuco_detector_and_board()
    squares_x, squares_y = 8, 22
    square_len = 14e-3
    marker_len = 10e-3

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_len, marker_len, aruco_dict)

    detector_params = cv2.aruco.DetectorParameters()
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

    return charuco_detector, board
end

function calibrate_naively(charuco_detector)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # kernel = np.ones((6,6), np.uint8)
    # eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    # closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(img)


    img = cv2.imread("../../calib/Camera30000.tif", cv2.IMREAD_GRAYSCALE)
    ids_collected, corners_collected = uniquely_identifiable_points_from_image_of_charuco_board(img, charuco_detector)

    for filepath in ["../../calib/Camera30001.tif","../../calib/Camera30002.tif","../../calib/Camera30003.tif","../../calib/Camera30004.tif",]
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        ids, corners = uniquely_identifiable_points_from_image_of_charuco_board(img, charuco_detector)

        ids_collected = vcat(ids_collected, ids)
        corners_collected = vcat(corners_collected, corners)
    end
    
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
        charuco_ids, (0,0,255))

    imshow(vis)
end

function imshow(img)
    cv2.imshow("image", img)
    while pyconvert(Int, cv2.waitKey(0)) != 27
        sleep(0.1)
    end
    cv2.destroyAllWindows()
end

function uniquely_identifiable_points_from_image_of_charuco_board(img, charuco_detector)
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(img)
    # board: 8x22
    # internal_corner_count = 21*7
    # marker_count = 4*22
    # marker_corner_count = 4*marker_count
    # total_corner_count = marker_corner_count + internal_corner_count

    ids::Vector{Int} = [pyconvert(Int, ind) for ind in charuco_ids]

    # for markerid in marker_ids
    #     # println("$markerid")
    #     markerid = pyconvert(Int64, markerid)
    #     distributed_ids = distribute_marker_corner_id(markerid)

    #     for did in distributed_ids
    #         push!(ids, did)
    #     end
    # end

    corners::Matrix{Float64} = pyconvert(Matrix{Float64}, charuco_corners)

    # for fourmarkercorners in marker_corners
    #     fourmarkercorners = pyconvert(Matrix{Float64}, fourmarkercorners[0])

    #     corners = vcat(corners, fourmarkercorners)
    # end
    return ids, corners
end

function distribute_marker_corner_id(id::Int64)
    return [1, 2, 3, 4].+(200 + id*4)
end

# Optimization.jl solve for system of equations
