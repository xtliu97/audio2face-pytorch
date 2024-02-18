from subprocess import call
from rich import progress
from tqdm import tqdm
import os
import time
from multiprocessing import Pool
import numpy as np
import pyrender
import trimesh
import cv2


from utils.facemesh import FaceMesh

verts = np.load("verts_sample.npy")
center = np.mean(verts[0], axis=0)
last_frame = verts[0]


def render(index):
    global verts, center, last_frame
    verts_sample = verts[index]
    texture_mesh = FaceMesh.load("FLAME_sample.obj")
    # rotate verts_sample
    rot = np.zeros(3)
    verts_sample[:] = cv2.Rodrigues(rot)[0].dot((verts_sample - center).T).T + center

    texture_mesh = texture_mesh.copy()
    texture_mesh.set_verts(verts_sample)

    # render
    render_mesh = pyrender.Mesh.from_trimesh(texture_mesh.trimesh(), smooth=True)

    camera_params = {
        "c": np.array([400, 400]),
        "k": np.array([-0.19816071, 0.92822711, 0, 0, 0]),
        "f": np.array([4754.97941935 / 2, 4754.97941935 / 2]),
    }

    frustum = {"near": 0.01, "far": 3.0, "height": 800, "width": 800}

    z_offset = 0.0
    intensity = 1.5

    scene = pyrender.Scene(ambient_light=[0.2, 0.2, 0.2], bg_color=[255, 255, 255])
    camera = pyrender.IntrinsicsCamera(
        fx=camera_params["f"][0],
        fy=camera_params["f"][1],
        cx=camera_params["c"][0],
        cy=camera_params["c"][1],
        znear=frustum["near"],
        zfar=frustum["far"],
    )

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 1.0 - z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3, 3]
    light_color = np.array([1.0, 1.0, 1.0])
    light = pyrender.PointLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3, 3] = pos
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES

    try:
        r = pyrender.OffscreenRenderer(
            viewport_width=frustum["width"], viewport_height=frustum["height"]
        )
        color, depth = r.render(scene, flags=flags)
        last_frame = color
    except Exception as e:
        print("pyrender: Failed rendering frame" + str(e))
        color = last_frame
    color = color[..., ::-1]
    cv2.imwrite(f"output/render_{index}.png", color)


if __name__ == "__main__":
    n = verts.shape[0]
    print(f"Rendering {n} images")

    os.makedirs("output", exist_ok=True)

    tic = time.time()
    for i in progress.track(range(n)):
        render(i)

    print(f"Rendered {n} images in {time.time() - tic} seconds")

    call(
        [
            "ffmpeg",
            "-framerate",
            "60",
            "-i",
            f"output/render_%d.png",
            "-c:v",
            "libx264",
            "-r",
            "30",
            "-pix_fmt",
            "yuv420p",
            "output.mp4",
        ]
    )
    # call(["rm", f"output/render_{i}.png"])
