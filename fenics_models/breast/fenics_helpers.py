import dolfinx
import numpy as np
from math import sin, cos, pi
import os
import itertools
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import pyvista
import tetridiv
import utils
import pyacvd
import tetgen


#
#   Mesh Creation Methods
#

def create_breast_mesh(mesh_dir, num_surfpts):
    breast_mesh = pyvista.read(mesh_dir)
    breast_mesh = breast_mesh.scale([1000, 1000, 1000], inplace=False)
    refined_mesh = _remesh_surface(breast_mesh, num_surfpts)
    min_pt = np.array(refined_mesh.bounds[::2])
    refined_mesh = refined_mesh.translate(-1*min_pt, inplace=False)
    refined_mesh = refined_mesh.rotate_y(90, inplace=False)
    hex_mesh = _hexahedralise_mesh(refined_mesh)
    return hex_mesh

def _remesh_surface(mesh, num_pts, num_subdiv=3):
    clus = pyacvd.Clustering(mesh)
    clus.subdivide(num_subdiv)
    clus.cluster(num_pts)
    remesh = clus.create_mesh()
    remesh = utils.create_pyvista_grid(remesh)
    return remesh

def _hexahedralise_mesh(mesh, mindihedral=20):
    tet = tetgen.TetGen(mesh)
    tet.tetrahedralize(order=1, mindihedral=mindihedral)
    tet_mesh = tet.grid
    hex_mesh = tetridiv.tet2hex(tet_mesh, output_type='dolfinx')
    return hex_mesh

#
#   Simulation Methods
#

def simulate_neohookean_breast(mesh, z_rot, y_rot, C_1, density, g, kappa, elem_order, rtol, atol, max_iter, x_to_fix, num_load_steps=None, u_guess=None, **ignored_kwargs):
    # Create function space:
    V = dolfinx.fem.VectorFunctionSpace(mesh, ("CG", elem_order))
    bcs = _create_clamped_bcs(x_to_fix, mesh, V)
    u = dolfinx.fem.Function(V)
    B = dolfinx.fem.Constant(mesh, [0.,0.,0.])
    F = _create_neohookean_constitutive_eqn(C_1, kappa, mesh, V, u, B)
    # Define problem:
    solver = _create_nonlinear_solver(F, u, V, bcs, rtol, atol, max_iter)
    f = _create_load_vector(z_rot, y_rot, density, g)
    u = _solve_nonlinear_system(solver, u, B, f, num_load_steps, u_guess)
    return u

def simulate_linear_breast(mesh, y_rot, x_rot, C_1, kappa, density, g, elem_order, rtol, atol, **ignored_kwargs):
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
    
def _create_clamped_bcs(x_to_fix, mesh, V):
    fixed = lambda x: x[0] <= x_to_fix
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

def _create_linear_material_problem(mesh, mu, lambda_, f, V, bc, atol, rtol):
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

def _create_load_vector(z_rot, y_rot, density, g, g_dir=(1,0,0)):
    f = density*g*np.array(g_dir)
    f = utils.rotate_vector(f, z_rot, y_rot)
    return f