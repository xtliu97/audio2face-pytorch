import os

import numpy as np
import trimesh


class FaceMesh:
    def __check_mesh_validity(self, arr):
        assert arr.ndim == 2, f"arr.ndim must be 2, got arr shape {arr.shape}"
        assert arr.shape[1] == 3, f"arr.shape[1] must be 3, got arr shape {arr.shape}"

    def __init__(self, verts, faces):
        self.__check_mesh_validity(verts)
        self.__check_mesh_validity(faces)

        self._verts = np.array(verts)
        self._faces = np.array(faces)

    @property
    def verts(self):
        return self._verts

    def set_verts(self, verts) -> None:
        self.__check_mesh_validity(verts)
        self._verts = np.array(verts)

    @property
    def faces(self):
        return self._faces

    @classmethod
    def load(cls, fname: str) -> "FaceMesh":
        assert os.path.exists(fname), f"{fname} does not exist"
        if fname.endswith(".obj"):
            trimesh_mesh = trimesh.load(fname)
            return cls(trimesh_mesh.vertices, trimesh_mesh.faces)
        elif fname.endswith(".ply"):
            try:
                from psbody.mesh import Mesh
            except ImportError:
                raise ImportError(
                    "psbody.mesh is required to load .ply file, \
if you don't have psbody.mesh, please use convert_ply_to_obj() in utils/convert_ply.py to convert .ply to .obj"
                )
            mesh = Mesh(filename=fname)
            return cls(mesh.v, mesh.f)

    def copy(self):
        return FaceMesh(self.verts.copy(), self.faces.copy())

    def trimesh(self):
        return trimesh.Trimesh(vertices=self.verts, faces=self.faces)
