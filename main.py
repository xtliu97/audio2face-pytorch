import numpy as np
import cv2

from utils.facemesh import FaceMesh
from utils.renderer import Renderer, images_to_video


if __name__ == "__main__":
    texture_mesh = FaceMesh.load("assets/FLAME_sample.obj")
    renderer = Renderer(texture_mesh)
    verts = np.load("assets/verts_sample.npy")
    rendered_images = renderer.render(verts)
    images_to_video(rendered_images, "output.mp4")