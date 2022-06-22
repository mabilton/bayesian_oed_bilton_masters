from math import cos, pi, sin

import numpy as np
import pyvista
import vtk

def create_param_combos(**kwargs):
    keys = kwargs.keys()
    param_combos = []
    for bundle in itertools.product(*kwargs.values()):
        param_dict = dict(zip(keys, bundle))
        param_combos.append(param_dict)
    return param_combos

#
#   Rotation Methods
#


def rotate_vector(f, z_rot, y_rot):
    # Using Euler angles - see https://www.autonomousrobotslab.com/frame-rotations-and-representations.html
    rot_matrix = create_rot_matrix(z_rot, y_rot)
    rotated_f = rot_matrix @ f
    return rotated_f


def create_rot_matrix(z_rot, y_rot, ang_to_rad=pi / 180):
    # NB: Negative associated with y so increasing y_rot goes in 'right direction'
    y_rot *= ang_to_rad
    z_rot *= ang_to_rad
    z_rot_matrix = np.array([ [cos(z_rot), sin(z_rot), 0],
                              [-sin(z_rot), cos(z_rot), 0],
                              [0, 0, 1]])
    y_rot_matrix = np.array([ [ cos(y_rot), 0, sin(y_rot) ],
                              [ 0, 1, 0 ],
                              [-sin(y_rot), 0, cos(y_rot)]])
    rot_matrix = y_rot_matrix @ z_rot_matrix
    return rot_matrix


#
#   Dolfinx Methods
#


def get_dolfinx_mesh_cells_at_query_points(query_points, mesh):
    import dolfinx

    bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.topology.dim)
    cells, points_on_processors = [], []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = dolfinx.geometry.compute_collisions(bb_tree, query_points)
    # Choose one of the cells that contains the point
    colliding_cells = dolfinx.geometry.compute_colliding_cells(
        mesh, cell_candidates, query_points
    )
    for i, point in enumerate(query_points):
        if len(colliding_cells.links(i)) > 0:
            points_on_processors.append(point)
            cells.append(colliding_cells.links(i)[0])
    return points_on_processors, cells


#
#   Mesh Conversion
#

_vtk_idx = {
    "hexahedron": vtk.VTK_HEXAHEDRON,
    "tetra": vtk.VTK_TETRA,
    "quad": vtk.VTK_QUAD,
    "triangle": vtk.VTK_TRIANGLE,
}


def create_pyvista_grid(mesh):
    mesh_class = str(type(mesh))
    if isinstance(mesh, dict):
        grid = _create_grid_from_dict(mesh)
    elif "dolfinx" in mesh_class:
        grid = _create_grid_from_dolfinx(mesh)
    elif "meshio" in mesh_class:
        grid = _create_grid_from_meshio(mesh)
    elif "pyvista" in mesh_class:
        grid = mesh
    elif "trimesh" in mesh_class:
        trimesh_dict = {"cell_geom": 'triangle', 'points': mesh.vertices, 'cells': mesh.faces}
        grid = _create_grid_from_dict(trimesh_dict)
    else:
        raise ValueError(
            "Invalid mesh type; must choose between dolfinx, meshio, or pyvista mesh."
        )
    return grid


def _create_grid_from_dict(mesh):
    vtk_idx = _vtk_idx[mesh["cell_geom"]]
    points = _ensure_points_3d(mesh["points"])
    return pyvista.UnstructuredGrid({vtk_idx: mesh["cells"]}, points)


def _create_grid_from_dolfinx(mesh):
    import dolfinx

    V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
    grid = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(V))
    return grid


def _create_grid_from_meshio(mesh):
    identified_mesh_type = False
    for cell_key, vtk_idx in _vtk_idx.items():
        if cell_key in mesh.cells_dict.keys():
            cells = np.array(mesh.cells_dict[cell_key], dtype=int)
            points = mesh.points
            if points.shape[-1] == 2:
                zeros = np.zeros((points.shape[0], 1))
                points = np.concatenate([points, zeros], axis=1)
            grid = pyvista.UnstructuredGrid({vtk_idx: cells}, points)
            identified_mesh_type = True
            break
    if not identified_mesh_type:
        raise ValueError("Unsupported meshio element type.")
    return grid


def _ensure_points_3d(points):
    if points.shape[-1] == 2:
        zeros = np.zeros((points.shape[0], 1))
        points = np.concatenate([points, zeros], axis=1)
    return points