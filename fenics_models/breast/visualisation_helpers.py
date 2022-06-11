import numpy as np
import pyvista
import utils
import vtk

#
#   Main Methods
#


def plot_mesh(
    mesh,
    cam_pos="xy",
    window_size=(960, 480),
    sargs=None,
    clim=None,
    n_labels=5,
    label_decimals=0,
    zoom=1,
    slice_mesh=False,
    points=None
):

    if sargs is None:
        sargs = _create_default_sargs(n_labels, label_decimals)

    grid = utils.create_pyvista_grid(mesh)

    pyvista.start_xvfb()
    p = pyvista.Plotter(window_size=window_size)
    if slice_mesh:
        cell_centres = grid.cell_centers().points
        mesh_centre = np.mean(cell_centres, axis=0)
        cells_to_remove = cell_centres[:, 0] >= mesh_centre[0]
        grid_rem = grid.remove_cells(cells_to_remove)
        grid_rem.clear_field_data()
        p.add_mesh(grid_rem, show_edges=True, color="white")
    else:
        p.add_mesh(grid, show_edges=True)
    
    if points is not None:
        pts = pyvista.PolyData(points)
        p.add_mesh(pts, color='red')
        
    p.show_axes()
    p.camera_position = cam_pos
    p.camera.zoom(zoom)
    return p


def plot_deformation(
    u,
    mesh,
    z_rot=0,
    y_rot=0,
    window_size=(960, 480),
    sargs=None,
    clim=None,
    n_labels=5,
    label_decimals=0,
    zoom=1,
    deform_factor=1,
    campos = 'yz'
):

    if "dolfinx" not in str(type(mesh)):
        raise ValueError(
            "Must provide dolfinx mesh corresponding to solved displacements u."
        )

    if sargs is None:
        sargs = _create_default_sargs(n_labels, label_decimals)

    grid = utils.create_pyvista_grid(mesh)
    
    # Interpolate deformations at nodes:
    points_on_processors, cells = utils.get_dolfinx_mesh_cells_at_query_points(
        grid.points, mesh
    )
    u_def = u.eval(points_on_processors, cells)
    grid.point_data["Deformation / mm"] = u_def

    pyvista.start_xvfb()
    p = pyvista.Plotter(window_size=window_size)
    # Show undeformed mesh as wireframe:
    p.add_mesh(grid, style="wireframe", color="k")
    # Plot deformed mesh:
    warped = grid.warp_by_vector("Deformation / mm", factor=deform_factor)
    p.add_mesh(warped, clim=clim, scalar_bar_args=sargs)
    p.show_axes()
    p.camera_position = campos
    p.camera.zoom(zoom)
    return p


#
#   Misc
#
    

def _create_default_sargs(n_labels, label_decimals):
    return dict(
        label_font_size=16,
        shadow=True,
        n_labels=n_labels,
        fmt=f"%.{label_decimals}f",
        font_family="arial",
    )
