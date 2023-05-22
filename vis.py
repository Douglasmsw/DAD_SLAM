import skimage.measure
import trimesh
import open3d as o3d
import numpy as np

def marching_cubes(occupancy, level=0.5):
    try:
        vertices, faces, vertex_normals, _ = skimage.measure.marching_cubes(    #marching_cubes_lewiner(    #marching_cubes(
            occupancy, level=level, gradient_direction='ascent') # extracts surface mesh from 3D volume
            # see https://scikit-image.org/docs/stable/auto_examples/edges/plot_marching_cubes.html for documentation
    except (RuntimeError, ValueError):
       return None

    dim = occupancy.shape[0]
    vertices = vertices / (dim - 1)
    print("Building trimesh")
    mesh = trimesh.Trimesh(vertices=vertices,
                           vertex_normals=vertex_normals,
                           faces=faces) # builds trimesh object 
                           # see https://trimsh.org/trimesh.html#trimesh.Trimesh for documentation

    return mesh

def trimesh_to_open3d(src):
    dst = o3d.geometry.TriangleMesh()
    dst.vertices = o3d.utility.Vector3dVector(src.vertices)
    dst.triangles = o3d.utility.Vector3iVector(src.faces)
    vertex_colors = src.visual.vertex_colors[:, :3].astype(np.float) / 255.0
    dst.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    dst.compute_vertex_normals()

    return dst