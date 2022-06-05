import dolfinx
import numpy as np
from math import sin, cos, pi
import os
import itertools
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import pyvista
    
def create_param_combos(**kwargs):
    keys = kwargs.keys()
    param_combos = []
    for bundle in itertools.product(*kwargs.values()):
        param_dict = dict(zip(keys, bundle))
        param_combos.append(param_dict)
    return param_combos

#
#   Visualisation Methods
#

def plot_deformation(u, mesh, beam_angle, cam_pos='xz', window_size=(960,480), rot_x=None, 
                     sargs=None, clim=None, n_labels=5, label_decimals=0, zoom=1, deform_factor=1):
    
    if sargs is None:
        sargs = dict(label_font_size=16,
                     shadow=True,
                     n_labels=n_labels,
                     fmt=f"%.{label_decimals}f",
                     font_family="arial")
    
    # Create VTK mesh:
    V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
    grid = pyvista.UnstructuredGrid(*dolfinx.plot.create_vtk_mesh(V))
    
    # Interpolate deformations at nodes:
    points_on_processors, cells = _get_mesh_cells_at_query_points(grid.points, mesh)
    u_def = u.eval(points_on_processors, cells)
    grid.point_data["Deformation / mm"] = u_def
    
    pyvista.start_xvfb()
    p = pyvista.Plotter()
    # Show undeformed mesh as wireframe:
    actor_0 = p.add_mesh(grid, style="wireframe", color="k")
    # Plot deformed mesh:
    warped = grid.warp_by_vector("Deformation / mm", factor=deform_factor)
    actor_1 = p.add_mesh(warped, clim=clim, scalar_bar_args=sargs)
    p.show_axes()
    p.camera_position = cam_pos
    p.camera.roll += beam_angle - 90
    p.camera.zoom(zoom)
    return p

#
#   Mesh Creation Methods
#

def create_cuboidal_mesh(L, W, NL, NW):
    mesh = dolfinx.mesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [L, W, W]], [NL, NW, NW], dolfinx.cpp.mesh.CellType.hexahedron)
    return mesh

def create_sliced_beam_mesh(W_1, W_2, L_1, L_2, mesh_size, num_layers):
    # Create meshio file:
    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon([[0.0, 0.0, 0.0],
                                 [0.0, 0.0, W_1],
                                 [L_2, 0.0, W_1],
                                 [L_1, 0.0, W_2],
                                 [L_1, 0.0, 0.0]], mesh_size=mesh_size)
        geom.extrude(poly, [0.0, W_1, 0.0], num_layers=num_layers)
        pygmsh_mesh = geom.generate_mesh(dim=3)
        pygmsh.write('beam_mesh.msh')
    # Also take note of number of elements in the mesh:
    num_elem = pygmsh_mesh.get_cells_type("tetra").shape[0]
    # Load mesh file:
    mesh3D_from_msh = meshio.read("beam_mesh.msh")
    cells = mesh3D_from_msh.get_cells_type("tetra")
    tetra_mesh = meshio.Mesh(points=mesh3D_from_msh.points, cells={"tetra": cells})
    # Convert to xdmf file then load:
    meshio.write("beam_mesh.xdmf", tetra_mesh)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "beam_mesh.xdmf", 'r') as f:
        mesh = f.read_mesh(name="Grid")
    os.system('rm beam_mesh.msh')
    os.system('rm beam_mesh.xdmf')
    return mesh, num_elem

#
#   Beam Simulation Methods
#

def simulate_neohookean_beam(mesh, beam_angle, C_1, density, g, kappa, elem_order, rtol, atol, max_iter, num_load_steps=None, u_guess=None, **ignored_kwargs):
    _clear_fenics_cache()
    # Create function space:
    V = dolfinx.fem.VectorFunctionSpace(mesh, ("CG", elem_order))
    bcs = _create_clamped_bcs(mesh, V)
    u = dolfinx.fem.Function(V)
    B = dolfinx.fem.Constant(mesh, [0.,0.,0.])
    F = _create_neohookean_constitutive_eqn(C_1, kappa, mesh, V, u, B)
    # Define problem:
    solver = _create_nonlinear_solver(F, u, V, bcs, rtol, atol, max_iter)
    f = _create_load_vector(beam_angle, density, g)
    u = _solve_nonlinear_system(solver, u, B, f, num_load_steps, u_guess)
    return u

def simulate_linear_beam(mesh, beam_angle, C_1, kappa, density, g, elem_order, rtol, atol, **ignored_kwargs):
    _clear_fenics_cache()
    V = dolfinx.fem.VectorFunctionSpace(mesh, ("CG", elem_order))
    bcs = _create_clamped_bcs(mesh, V)
    mu = 2*C_1
    lambda_ = 2*kappa
    f = _create_load_vector(beam_angle, density, g)
    problem = _create_linear_beam_problem(mesh, mu, lambda_, f, V, bcs, atol, rtol)
    u = problem.solve()
    return u

#
#   Mesh Deformation Post-Processing Methods
#

# How to evaluate solution at specific (potentially non-nodel) point: https://jorgensd.github.io/dolfinx-tutorial/chapter1/membrane_code.html
def compute_end_displacement(u, mesh, W, L):
    end_point = np.array([[L, W, W]])
    points_on_processors, cells = _get_mesh_cells_at_query_points(end_point, mesh)
    u_end = u.eval(points_on_processors, cells)
    disp = np.sum(u_end**2)**(1/2)
    return disp.item()

def _get_mesh_cells_at_query_points(query_points, mesh):
    bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.topology.dim)
    cells, points_on_processors = [], []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = dolfinx.geometry.compute_collisions(bb_tree, query_points)
    # Choose one of the cells that contains the point
    colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, query_points)
    for i, point in enumerate(query_points):
        if len(colliding_cells.links(i))>0:
            points_on_processors.append(point)
            cells.append(colliding_cells.links(i)[0])
    return points_on_processors, cells

def compute_pre_and_postdeformation_volume(u, mesh, quad_order=4):
    ndim = mesh.geometry.x.shape[1]
    I = ufl.Identity(ndim)
    dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 4})
    const_funspace = dolfinx.fem.VectorFunctionSpace(mesh, ("DG", 0), dim=1)
    const_fun = dolfinx.fem.Function(const_funspace)
    const_fun.vector[:] = np.ones(const_fun.vector[:].shape)
    vol_before = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(const_fun,const_fun)*dx))
    F = I + ufl.grad(u)
    vol_after = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.det(F)*dx))
    return [vol_before, vol_after]

#
#   NeoBeam Simulation Helper Methods
#

def _clear_fenics_cache(cache_dir='/root/.cache/fenics'):
    if os.path.isdir(cache_dir) and len(os.listdir(cache_dir))>0:
        os.system(f'rm -r {cache_dir}/*')
    
def _create_clamped_bcs(mesh, V):
    fixed = lambda x: np.isclose(x[0], 0)
    fixed_facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, fixed)
    facet_tag = dolfinx.mesh.meshtags(mesh, mesh.topology.dim-1, fixed_facets, 1)
    u_bc = dolfinx.fem.Function(V)
    with u_bc.vector.localForm() as loc:
        loc.set(0)
    left_dofs = dolfinx.fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.indices[facet_tag.values==1])
    bcs = [dolfinx.fem.dirichletbc(u_bc, left_dofs)]
    return bcs   

def _create_nonlinear_solver(F, u, V, bcs, rtol, atol, max_iter):
    problem = dolfinx.fem.petsc.NonlinearProblem(F, u, bcs)
    solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
    solver.rtol = rtol
    solver.atol = atol
    solver.max_it = max_iter
    return solver
        
def _create_neohookean_constitutive_eqn(C_1, kappa, mesh, V, u, B, quad_degree=4):
    v = ufl.TestFunction(V)
    d = len(u)
    I = ufl.variable(ufl.Identity(d))
    F = ufl.variable(I + ufl.grad(u))
    C = ufl.variable(F.T * F)
    J = ufl.variable(ufl.det(F))
    Ic = ufl.variable(ufl.tr(C))
    # Nearly-Incompressible Neo-Hookean material; 
    # See: https://link.springer.com/article/10.1007/s11071-015-2167-1
    psi = C_1*(Ic-3) + kappa*(J-1)**2
    P = ufl.diff(psi, F)
    metadata = {"quadrature_degree": 4}
    dx = ufl.Measure("dx", metadata=metadata)
    Pi = ufl.inner(ufl.grad(v), P)*dx - ufl.inner(v, B)*dx
    return Pi

def _solve_nonlinear_system(solver, u, B, f, num_load_steps, u_guess):
    if (num_load_steps is None) and (u_guess is None):
        raise ValueError('Must specify either number of load steps of an initial guess for u.')
    if u_guess is not None:
        u.x.array[:] = u_guess.x.array[:]
        num_its, converged = solver.solve(u)
        if not converged:
            raise ValueError(f'Failed to converge.')
    # Load stepping
    else:
        for step_i in range(num_load_steps):
            print(f'Performing load step {step_i+1}/{num_load_steps}...')
            for j, f_j in enumerate(f):
                B.value[j] = ((step_i+1)/num_load_steps)*f_j
            num_its, converged = solver.solve(u)
            if not converged:
                raise ValueError(f'Load step {step_i+1} failed to converge.')
            u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    return u

def _create_linear_beam_problem(mesh, mu, lambda_, f, V, bc, atol, rtol):
    u_D = dolfinx.fem.Function(V)
    with u_D.vector.localForm() as loc:
        loc.set(0)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    epsilon = lambda u: ufl.sym(ufl.grad(u))
    sigma = lambda u: lambda_ * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) + 2*mu*epsilon(u)
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    f_const = dolfinx.fem.Constant(mesh, f)
    L = ufl.dot(f_const, v) * ufl.dx 
    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=bc, petsc_options={"ksp_atol":atol, "ksp_rtol":rtol, "ksp_type": "preonly", "pc_type": "lu"})
    return problem

#
#  Loading and Rotation Methods
#

def _create_load_vector(beam_angle, density, g, g_dir=(1,0,0)):
    f = density*g*np.array(g_dir)
    f = _rotate_vector(f, y_rot=beam_angle)
    return f
    
def _rotate_vector(f, y_rot, x_rot=0):
    # Using Euler angles - see https://www.autonomousrobotslab.com/frame-rotations-and-representations.html
    rot_matrix = _create_rot_matrix(y_rot, x_rot)
    rotated_f = rot_matrix @ f
    return rotated_f

def _create_rot_matrix(y_rot, x_rot, ang_to_rad=pi/180):
    # NB: Negative associated with y so increasing y_rot goes in 'right direction'
    theta, psi = -ang_to_rad*y_rot, ang_to_rad*x_rot
    rot_matrix = np.array([[         cos(theta),        0,          -sin(theta)],
                           [sin(psi)*sin(theta),  cos(psi), sin(psi)*cos(theta)],
                           [cos(psi)*sin(theta), -sin(psi), cos(psi)*cos(theta)]])
    return rot_matrix