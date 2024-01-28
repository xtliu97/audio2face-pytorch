from functools import cached_property
import time

import pyrender
import numpy as np
import cv2
from rich.progress import track


from .facemesh import FaceMesh


class Renderer:
    def __init__(self, texture_mesh: FaceMesh):
        self.texture_mesh = texture_mesh
        self.camera_params = {
            "c": np.array([400, 400]),
            "k": np.array([-0.19816071, 0.92822711, 0, 0, 0]),
            "f": np.array([4754.97941935 / 2, 4754.97941935 / 2]),
        }
        self.frustum = {"near": 0.01, "far": 3.0, "height": 800, "width": 800}
        self.z_offset = 0.0
        self.intensity = 1.5

    def _render(self, verts, queue=None):
        scene = self.scene
        render_mesh = self.texture_mesh.copy()
        render_mesh.set_verts(verts)
        render_mesh = pyrender.Mesh.from_trimesh(render_mesh.trimesh(), smooth=True)
        scene.add(render_mesh, pose=np.eye(4))

        renderer = self.renderer
        rendered_image, _ = renderer.render(
            scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES
        )

        if queue is not None:
            queue.put(rendered_image)

        return rendered_image

    @cached_property
    def scene(self) -> "pyrender.Scene":
        scene = pyrender.Scene(ambient_light=[0.2, 0.2, 0.2], bg_color=[255, 255, 255])
        camera = pyrender.IntrinsicsCamera(
            fx=self.camera_params["f"][0],
            fy=self.camera_params["f"][1],
            cx=self.camera_params["c"][0],
            cy=self.camera_params["c"][1],
            znear=self.frustum["near"],
            zfar=self.frustum["far"],
        )
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, 0, 1.0 - self.z_offset])
        scene.add(camera, pose=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])

        angle = np.pi / 6.0
        pos = camera_pose[:3, 3]

        light_color = np.array([1.0, 1.0, 1.0])
        light = pyrender.PointLight(color=light_color, intensity=self.intensity)

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

        return scene

    @cached_property
    def renderer(self):
        return pyrender.OffscreenRenderer(
            viewport_width=self.frustum["width"], viewport_height=self.frustum["height"]
        )

    def render(self, target_verts: np.ndarray):
        n_frames = target_verts.shape[0]
        tic = time.time()
        print(f"Rendering {n_frames} frames...")
        prev_rendered_image = None
        rendered_images = []
        n_success = 0
        for idx in track(range(n_frames)):
            try:
                rendered_image = self._render(target_verts[idx])
                n_success += 1
            except Exception as e:
                print("Failed rendering frame" + str(e))
                rendered_image = prev_rendered_image
            finally:
                prev_rendered_image = rendered_image
                rendered_images.append(rendered_image)
        toc = time.time()
        print(f"Rendering took {toc - tic:.2f}s, success  {n_success} / {n_frames}")
        return rendered_images
