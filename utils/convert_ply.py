# concert FLAME_sample.ply to FLAME_sample.obj of trimesh, in case of psbody is not installed
# However, this script requires psbody.mesh
import os
import trimesh
from psbody.mesh import Mesh


def convert_ply_to_obj(ply_fanme: str) -> None:
    base_dir = os.path.dirname(ply_fanme)
    obj_fname = os.path.join(
        base_dir, os.path.basename(ply_fanme).replace(".ply", ".obj")
    )
    mesh = Mesh(filename=ply_fanme)
    print(
        f"psbody.mesh.Mesh loaded from {ply_fanme}, vshape:{mesh.v.shape}, fshape:{mesh.f.shape}"
    )
    print("Converting to trimesh.Trimesh...")
    trimesh.Trimesh(vertices=mesh.v, faces=mesh.f).export(obj_fname)
    print("Saved to {}".format(obj_fname))


convert_ply_to_obj("../template/FLAME_sample.ply")
