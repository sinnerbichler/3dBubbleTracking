ROOT
├── cameras: object                         REQUIRED
│   └── <camera_id>: object                 REQUIRED
│       ├── position: [x, y, z]             REQUIRED
│       ├── rotation_euler_deg: [rx, ry, rz] OPTIONAL
│       └── rotation_euler_rad: [rx, ry, rz] OPTIONAL
│
├── markers: object                          OPTIONAL
│   └── <marker_id>: object
│
└── frames: array                            REQUIRED
    └── frame object
        ├── frame: integer                   REQUIRED
        │
        ├── detections: object               OPTIONAL
        │   └── <marker_id>: object
        │       └── <camera_id>: object
        │           └── ray: array[3][3]     REQUIRED
        │               ├── [camera position]
        │               ├── [water interface]
        │               └── [ray termination]
        │
        └── triangulations: object           OPTIONAL
            └── <marker_id>: object
                ├── pairs: object            OPTIONAL
                │   ├── "0-1": [x,y,z]
                │   ├── "0-2": [x,y,z]
                │   ├── "0-3": [x,y,z]
                │   ├── "1-2": [x,y,z]
                │   ├── "1-3": [x,y,z]
                │   └── "2-3": [x,y,z]
                │
                └── mean: [x,y,z]            OPTIONAL


{
  "cameras": {
    "0": {
      "position": [2.0, -1.5, 1.2],
      "rotation_euler_deg": [70.0, 0.0, 35.0]
    },
    "1": {
      "position": [2.0, 1.5, 1.2],
      "rotation_euler_deg": [70.0, 0.0, 145.0]
    },
    "2": {
      "position": [-2.0, 1.5, 1.2],
      "rotation_euler_deg": [70.0, 0.0, 215.0]
    },
    "3": {
      "position": [-2.0, -1.5, 1.2],
      "rotation_euler_deg": [70.0, 0.0, 325.0]
    }
  },

  "markers": {
    "0": {},
    "1": {}
  },

  "frames": [
    {
      "frame": 0,

      "detections": {
        "0": {
          "0": {
            "ray": [
              [2.0, -1.5, 1.2],
              [0.8, -0.2, 0.5],
              [0.4, 0.1, 0.3]
            ]
          },
          "1": {
            "ray": [
              [2.0, 1.5, 1.2],
              [0.8, 0.2, 0.5],
              [0.4, 0.1, 0.3]
            ]
          },
          "2": {
            "ray": [
              [-2.0, 1.5, 1.2],
              [0.1, 0.5, 0.5],
              [0.4, 0.1, 0.3]
            ]
          }
        },

        "1": {
          "0": {
            "ray": [
              [2.0, -1.5, 1.2],
              [0.7, -0.3, 0.6],
              [0.2, 0.2, 0.4]
            ]
          },
          "3": {
            "ray": [
              [-2.0, -1.5, 1.2],
              [0.0, -0.4, 0.6],
              [0.2, 0.2, 0.4]
            ]
          }
        }
      },

      "triangulations": {
        "0": {
          "pairs": {
            "0-1": [0.40, 0.10, 0.30],
            "0-2": [0.41, 0.09, 0.31],
            "1-2": [0.39, 0.11, 0.29]
          },
          "mean": [0.40, 0.10, 0.30]
        },

        "1": {
          "pairs": {
            "0-3": [0.20, 0.20, 0.40]
          },
          "mean": [0.20, 0.20, 0.40]
        }
      }
    },

    {
      "frame": 1,

      "detections": {},

      "triangulations": {}
    }
  ]
}
