import json
import math

import bpy
from mathutils import Vector

# ============================================================
# CONFIGURATION
# ============================================================

JSON_FILE = "/home/simon/mega/masterarbeit/code/generated-files/calibration.json"

# Blender frame numbering.
# If your JSON starts at frame 0, this can remain 1 if you
# want Blender frame 1 to correspond to JSON frame 0.
BLENDER_FRAME_OFFSET = 0

# Visual sizes
RAY_RADIUS = 0.001
POINT_RADIUS = 0.002

# Camera display size
CAMERA_DISPLAY_SIZE = 0.10


# ============================================================
# LOAD JSON
# ============================================================

with open(JSON_FILE, "r") as f:
    DATA = json.load(f)


# ============================================================
# MATERIALS
# ============================================================

def make_material(name, color, alpha=1.0):
    mat = bpy.data.materials.get(name)

    if mat is None:
        mat = bpy.data.materials.new(name)

    mat.diffuse_color = (
        color[0],
        color[1],
        color[2],
        alpha
    )

    mat.use_nodes = True

    bsdf = mat.node_tree.nodes.get("Principled BSDF")

    if bsdf:
        bsdf.inputs["Base Color"].default_value = (
            color[0],
            color[1],
            color[2],
            1.0
        )

        bsdf.inputs["Roughness"].default_value = 0.4

        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = alpha

    if alpha < 1.0:
        mat.surface_render_method = 'DITHERED'

    return mat


CAMERA_MATERIAL = make_material(
    "Camera",
    (0.15, 0.15, 0.15)
)

RAY_AIR_MATERIALS = [
    make_material("Ray_Camera_0_Air", (1.0, 0.1, 0.1), 0.75),
    make_material("Ray_Camera_1_Air", (0.1, 1.0, 0.1), 0.75),
    make_material("Ray_Camera_2_Air", (0.1, 0.4, 1.0), 0.75),
    make_material("Ray_Camera_3_Air", (1.0, 0.7, 0.05), 0.75),
]

RAY_WATER_MATERIALS = [
    make_material("Ray_Camera_0_Water", (1.0, 0.2, 0.2), 0.90),
    make_material("Ray_Camera_1_Water", (0.2, 1.0, 0.2), 0.90),
    make_material("Ray_Camera_2_Water", (0.2, 0.5, 1.0), 0.90),
    make_material("Ray_Camera_3_Water", (1.0, 0.8, 0.1), 0.90),
]

POINT_PAIR_MATERIAL = make_material(
    "Triangulated_Point",
    (1.0, 0.15, 0.05)
)

POINT_MEAN_MATERIAL = make_material(
    "Mean_Point",
    (1.0, 1.0, 1.0)
)


# ============================================================
# COLLECTIONS
# ============================================================

def get_or_create_collection(name):
    collection = bpy.data.collections.get(name)

    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)

    return collection


CAMERA_COLLECTION = get_or_create_collection("Cameras")
RAY_COLLECTION = get_or_create_collection("Rays")
POINT_COLLECTION = get_or_create_collection("Triangulated Points")


# ============================================================
# HELPERS
# ============================================================

def hide_object(obj):
    obj.hide_viewport = True
    obj.hide_render = True


def show_object(obj):
    obj.hide_viewport = False
    obj.hide_render = False


def set_curve_endpoints(obj, p0, p1):
    """
    Update a two-point curve in world coordinates.
    """

    spline = obj.data.splines[0]

    spline.points[0].co = (
        p0[0],
        p0[1],
        p0[2],
        1.0
    )

    spline.points[1].co = (
        p1[0],
        p1[1],
        p1[2],
        1.0
    )


def create_ray(name, material):
    """
    Create a two-point 3D curve.
    """

    curve = bpy.data.curves.new(
        name=name,
        type='CURVE'
    )

    curve.dimensions = '3D'
    curve.resolution_u = 1
    curve.bevel_depth = RAY_RADIUS
    curve.bevel_resolution = 2

    spline = curve.splines.new('POLY')
    spline.points.add(1)

    spline.points[0].co = (0, 0, 0, 1)
    spline.points[1].co = (0, 0, 0, 1)

    obj = bpy.data.objects.new(name, curve)

    RAY_COLLECTION.objects.link(obj)

    obj.data.materials.append(material)

    hide_object(obj)

    return obj


def create_point(name, material):
    """
    Create a small UV sphere used for a triangulated point.
    """

    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=12,
        ring_count=6,
        radius=POINT_RADIUS,
        location=(0, 0, 0)
    )

    obj = bpy.context.object
    obj.name = name

    # Move to our collection.
    for collection in list(obj.users_collection):
        collection.objects.unlink(obj)

    POINT_COLLECTION.objects.link(obj)

    obj.data.materials.append(material)

    hide_object(obj)

    return obj


# ============================================================
# CAMERAS
# ============================================================

def create_camera(camera_id, camera_data):
    name = f"Camera_{camera_id}"

    cam_data = bpy.data.cameras.new(name)

    cam_obj = bpy.data.objects.new(name, cam_data)

    CAMERA_COLLECTION.objects.link(cam_obj)

    position = camera_data["position"]

    cam_obj.location = Vector(position)

    if "rotation_euler_deg" in camera_data:

        # rotation = camera_data["rotation_euler_deg"]

        # cam_obj.rotation_euler = (
        #     math.radians(rotation[0]),
        #     math.radians(rotation[1]),
        #     math.radians(rotation[2])
        # )
        raise ValueError("euler angles are broken!")

    elif "rotation_euler_rad" in camera_data:

        # rotation = camera_data["rotation_euler_rad"]

        # cam_obj.rotation_euler = (
        #     rotation[0],
        #     rotation[1],
        #     rotation[2]
        # )
        raise ValueError("euler angles are broken!")
    elif "rotation_quaternion" in camera_data:
        cam_obj.rotation_mode = "QUATERNION"
        quaternions = camera_data["rotation_quaternion"]
        cam_obj.rotation_quaternion = quaternions

    cam_data.display_size = CAMERA_DISPLAY_SIZE

    return cam_obj


CAMERAS = {}

for camera_id, camera_data in DATA["cameras"].items():

    CAMERAS[camera_id] = create_camera(
        camera_id,
        camera_data
    )


# ============================================================
# CREATE STATIC VISUALIZATION OBJECTS
# ============================================================

MARKER_IDS = list(DATA.get("markers", {}).keys())

# If markers aren't explicitly listed, discover them from frames.
if not MARKER_IDS:

    marker_ids = set()

    for frame in DATA.get("frames", []):

        marker_ids.update(
            frame.get("detections", {}).keys()
        )

        marker_ids.update(
            frame.get("triangulations", {}).keys()
        )

    MARKER_IDS = sorted(marker_ids)


CAMERA_IDS = list(DATA["cameras"].keys())
print("CAMERA_IDS: ", CAMERA_IDS)


# ------------------------------------------------------------
# Rays
# ------------------------------------------------------------

# Dictionary:
#
# rays[marker_id][camera_id]["air"]
# rays[marker_id][camera_id]["water"]
#
RAYS = {}


for marker_id in MARKER_IDS:

    RAYS[marker_id] = {}

    for camera_id in CAMERA_IDS:

        air_name = (
            f"Ray_M{marker_id}_C{camera_id}_Air"
        )

        water_name = (
            f"Ray_M{marker_id}_C{camera_id}_Water"
        )

        camera_index = int(camera_id)-1

        air = create_ray(
            air_name,
            RAY_AIR_MATERIALS[camera_index]
        )

        water = create_ray(
            water_name,
            RAY_WATER_MATERIALS[camera_index]
        )

        RAYS[marker_id][camera_id] = {
            "air": air,
            "water": water
        }


# ------------------------------------------------------------
# Triangulated points
# ------------------------------------------------------------

POINTS = {}

PAIR_NAMES = [
    "0-1",
    "0-2",
    "0-3",
    "1-2",
    "1-3",
    "2-3"
]


for marker_id in MARKER_IDS:

    POINTS[marker_id] = {}

    for pair in PAIR_NAMES:

        name = (
            f"Point_M{marker_id}_Pair_{pair}"
        )

        POINTS[marker_id][pair] = create_point(
            name,
            POINT_PAIR_MATERIAL
        )

    mean_name = f"Point_M{marker_id}_Mean"

    POINTS[marker_id]["mean"] = create_point(
        mean_name,
        POINT_MEAN_MATERIAL
    )


# ============================================================
# FRAME UPDATE
# ============================================================

def hide_all_dynamic_objects():
    """
    Used at the beginning of every frame.
    This makes empty frames and missing detections safe.
    """

    for marker_id in RAYS:

        for camera_id in RAYS[marker_id]:

            hide_object(
                RAYS[marker_id][camera_id]["air"]
            )

            hide_object(
                RAYS[marker_id][camera_id]["water"]
            )

    for marker_id in POINTS:

        for point_name in POINTS[marker_id].values():

            hide_object(point_name)


def update_frame(frame_data):

    # First make absolutely everything invisible.
    hide_all_dynamic_objects()

    detections = frame_data.get(
        "detections",
        {}
    )

    triangulations = frame_data.get(
        "triangulations",
        {}
    )

    # --------------------------------------------------------
    # RAYS
    # --------------------------------------------------------

    for marker_id, camera_detections in detections.items():

        if marker_id not in RAYS:
            continue

        for camera_id, detection in camera_detections.items():

            if camera_id not in RAYS[marker_id]:
                continue

            ray_points = detection.get("ray")

            if not ray_points:
                continue

            if len(ray_points) != 3:
                print(
                    f"WARNING: marker {marker_id}, "
                    f"camera {camera_id}: "
                    f"ray does not contain 3 points"
                )
                continue

            camera_point = ray_points[0]
            interface_point = ray_points[1]
            termination_point = ray_points[2]

            air_ray = RAYS[marker_id][camera_id]["air"]
            water_ray = RAYS[marker_id][camera_id]["water"]

            set_curve_endpoints(
                air_ray,
                camera_point,
                interface_point
            )

            set_curve_endpoints(
                water_ray,
                interface_point,
                termination_point
            )

            show_object(air_ray)
            show_object(water_ray)

    # --------------------------------------------------------
    # TRIANGULATED POINTS
    # --------------------------------------------------------

    for marker_id, triangulation in triangulations.items():

        if marker_id not in POINTS:
            continue

        pairs = triangulation.get(
            "pairs",
            {}
        )

        for pair_name, position in pairs.items():

            if pair_name not in POINTS[marker_id]:
                continue

            point_obj = POINTS[marker_id][pair_name]

            point_obj.location = Vector(position)

            show_object(point_obj)

        mean_position = triangulation.get("mean")

        if mean_position is not None:

            mean_obj = POINTS[marker_id]["mean"]

            mean_obj.location = Vector(
                mean_position
            )

            show_object(mean_obj)


# ============================================================
# CREATE FRAME DATA LOOKUP
# ============================================================

FRAMES = {}

for frame_data in DATA.get("frames", []):

    frame_number = frame_data["frame"]

    FRAMES[frame_number] = frame_data


# ============================================================
# ANIMATION HANDLER
# ============================================================

def frame_change_handler(scene):

    json_frame = (
        scene.frame_current
        - BLENDER_FRAME_OFFSET
    )

    frame_data = FRAMES.get(json_frame)

    if frame_data is None:

        # This is deliberately safe for empty/missing frames.
        hide_all_dynamic_objects()

        return

    update_frame(frame_data)


# Remove our handler if the script is run multiple times.
for handler in list(
    bpy.app.handlers.frame_change_pre
):

    if getattr(
        handler,
        "__name__",
        ""
    ) == "frame_change_handler":

        bpy.app.handlers.frame_change_pre.remove(
            handler
        )


bpy.app.handlers.frame_change_pre.append(
    frame_change_handler
)


# ============================================================
# SET FRAME RANGE
# ============================================================

json_frame_numbers = list(FRAMES.keys())

if json_frame_numbers:

    scene = bpy.context.scene

    scene.frame_start = (
        min(json_frame_numbers)
        + BLENDER_FRAME_OFFSET
    )

    scene.frame_end = (
        max(json_frame_numbers)
        + BLENDER_FRAME_OFFSET
    )


# ============================================================
# INITIAL FRAME
# ============================================================

if FRAMES:

    first_frame = min(FRAMES.keys())

    bpy.context.scene.frame_set(
        first_frame + BLENDER_FRAME_OFFSET
    )

else:

    hide_all_dynamic_objects()


print("Visualization setup complete.")
print(f"Cameras: {len(CAMERAS)}")
print(f"Markers: {len(MARKER_IDS)}")
print(f"Frames:  {len(FRAMES)}")
