import dolfinx
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from math import sin, cos, pi
import os
import itertools
# import pygmsh

def create_param_combos(**kwargs):
    keys = kwargs.keys()
    param_combos = []
    for bundle in itertools.product(*kwargs.values()):
        param_dict = dict(zip(keys, bundle))
        param_combos.append(param_dict)
    return param_combos

#
#   Mesh Creation Methods
#

def create_cuboidal_mesh(L, W, NL, NW):
    mesh = dolfinx.BoxMesh(MPI.COMM_WORLD,[[0.0, 0.0, 0.0], [L, W, W]], [NL, NW, NW], dolfinx.cpp.mesh.CellType.hexahedron)
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

# elem_order=2, rtol=1e-3, atol=1e-3, max_it=10
def simulate_neohookean_beam(mesh, beam_angle, C_1, density, g, kappa, num_load_steps, elem_order, rtol, atol, max_iter, **ignored_kwargs):
    # Create function space:
    V = dolfinx.VectorFunctionSpace(mesh, ("CG", elem_order))
    bcs = _create_clamped_bcs(mesh, V)
    F, u, B = _create_neohookean_constitutive_eqn(C_1, kappa, mesh, V)
    _clear_fenics_cache()
    # Define problem:
    problem = dolfinx.fem.NonlinearProblem(F, u, bcs)
    solver = _create_nonlinear_solver(problem, rtol, atol, max_iter)
    f = _create_load_vector(beam_angle, density, g)
    u = _perform_load_stepping(solver, u, B, f, num_load_steps)
    return u

def simulate_linear_beam(mesh, beam_angle, C_1, lambda_, density, g, elem_order, rtol, atol, **ignored_kwargs):
    _clear_model_cache()
    V = dolfinx.VectorFunctionSpace(mesh, ("CG", elem_order))
    bcs = _create_clamped_bcs(mesh, V)
    mu = 2*C_1
    f = _create_load_vector(beam_angle, density, g)
    problem = _create_linear_beam_problem(mesh, mu, lambda_, f, V, atol, rtol)
    u = problem.solve()
    return u

#
#   Mesh Deformation Post-Processing Methods
#

def compute_end_displacement(u, mesh, W, L):
    u_vals = u.compute_point_values().real
    idx = np.isclose(mesh.geometry.x, [L, W, W])
    idx = np.where(np.all(idx, axis=1))
    u_vals = u_vals[idx]
    disp = np.sum(u_vals**2, axis=1)**(1/2)
    return disp.item()

def compute_pre_and_postdeformation_volume(u, mesh, quad_order=4):
    ndim = mesh.geometry.x.shape[1]
    I = ufl.Identity(ndim)
    dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": quad_order})
    const_funspace = dolfinx.VectorFunctionSpace(mesh, ("DG", 0), dim=1)
    const_fun = dolfinx.Function(const_funspace)
    const_fun.vector[:] = np.ones(const_fun.vector[:].shape)
    ufl.inner(const_fun,const_fun)
    vol_before = dolfinx.fem.assemble.assemble_scalar(ufl.inner(const_fun,const_fun)*dx)
    F = I + ufl.grad(u)
    vol_after = dolfinx.fem.assemble.assemble_scalar(ufl.det(F)*dx)
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
    facet_tag = dolfinx.MeshTags(mesh, mesh.topology.dim-1, fixed_facets, 1)
    u_bc = dolfinx.Function(V)
    with u_bc.vector.localForm() as loc:
        loc.set(0)
    left_dofs = dolfinx.fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.indices[facet_tag.values==1])
    bcs = [dolfinx.DirichletBC(u_bc, left_dofs)]
    return bcs   

def _create_nonlinear_solver(problem, rtol, atol, max_iter):
    solver = dolfinx.NewtonSolver(MPI.COMM_WORLD, problem)
    solver.rtol = rtol
    solver.atol = atol
    solver.max_it = max_iter
    return solver

def _create_neohookean_constitutive_eqn(C_1, kappa, mesh, V, quad_degree=4):
    B = dolfinx.Constant(mesh, (0,0,0))
    v = ufl.TestFunction(V)
    u = dolfinx.Function(V)
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
    F = ufl.inner(ufl.grad(v), P)*dx - ufl.inner(v, B)*dx
    return F, u, B

def _perform_load_stepping(solver, u, B, f, num_load_steps):
    for step_i in range(num_load_steps):
        print(step_i)
        for j, f_j in enumerate(f):
            B.value[j] = ((step_i+1)/num_load_steps)*f_j
        num_its, converged = solver.solve(u)
        if not converged:
            raise ValueError(f'Load step {step_i+1}/{num_load_steps} failed to converge.')
        u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    return u

# "ksp_atol":1e-3, "ksp_rtol":1e-3
def _create_linear_beam_problem(mesh, mu, lambda_, f, V,  atol, rtol):
    u_D = dolfinx.Function(V)
    with u_D.vector.localForm() as loc:
        loc.set(0)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    epsilon = lambda u: ufl.sym(ufl.grad(u))
    sigma = lambda u: lambda_ * ufl.nabla_div(u) * ufl.Identity(u.geometric_dimension()) + 2*mu*epsilon(u)
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    f_const = dolfinx.Constant(mesh, (f[0], f[1], f[2]))
    L = ufl.dot(f_const, v) * ufl.dx 
    problem = dolfinx.fem.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_atol":atol, "ksp_rtol":rtol, "ksp_type": "preonly", "pc_type": "lu"})
    return problem

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