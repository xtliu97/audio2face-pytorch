import numpy as np

from utils.facemesh import FaceMesh
from utils import img_to_video

if __name__ == "__main__":
    from utils.renderer import Renderer

    texture_mesh = FaceMesh.load("assets/FLAME_sample.obj")
    renderer = Renderer(texture_mesh)

    verts = np.load("assets/verts_sample.npy")
    rendered_images = renderer.render(verts[:10])
    img_to_video(rendered_images, "output.mp4")

    # call(
    #     [
    #         "ffmpeg",
    #         "-framerate",
    #         "60",
    #         "-i",
    #         f"output/render_%d.png",
    #         "-c:v",
    #         "libx264",
    #         "-r",
    #         "30",
    #         "-pix_fmt",
    #         "yuv420p",
    #         "output.mp4",
    #     ]
    # )
    # # call(["rm", f"output/render_{i}.png"])
