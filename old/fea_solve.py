from mpi4py import MPI
from dolfinx import mesh
from dolfinx.fem import FunctionSpace
from dolfinx import fem
import numpy as np
import ufl
from petsc4py.PETSc import ScalarType

from dolfinx import geometry


N_elem = 128

PI = np.pi


# def u_true(x_):
#     x = x_[0]
#     y = x_[1]

#     return ufl.sin(x * PI) * ufl.sin(y * PI)


def eval_func(u, x, domain):

    print("Collecting cells")
    bb_tree = geometry.BoundingBoxTree(domain, domain.topology.dim)

    # where x is (N x 2)
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = geometry.compute_collisions(bb_tree, x.T)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, x.T)
    for i, point in enumerate(x.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    print(len(points_on_proc))
    print(len(cells))
    print(x.shape)

    print("Evaluating u")
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    u_values = u.eval(points_on_proc, cells)

    return u_values


def u_true_np(x_):
    x = x_[0]
    y = x_[1]

    return 1 + x**2 + 2 * y**2 + x**2 * y**2


def f_driving(x_):
    x = x_[0]
    y = x_[1]

    return -6 + 0 * x * y
    # return -2 * PI**2 * u_true_np(x_)


def solve_FEA(N_elem, u_true, f_driving, XY_eval):
    # create a N_elem x N_elem quad mesh (where N_elem is number of elements, not nodes)
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N_elem, N_elem, mesh.CellType.quadrilateral)

    # create function space, trial, and test functions
    V = FunctionSpace(domain, ("CG", 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # interpolate BCs onto function space
    uD = fem.Function(V)
    uD.interpolate(u_true)

    print(uD.x.array)

    # find dofs in mesh, create bc term
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)

    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs)

    # set driving coeff
    f = fem.Function(V)
    f.interpolate(f_driving)

    print(f.x.array)

    # set up weak form equations

    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    # set up solver
    problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    # check results
    V2 = fem.FunctionSpace(domain, ("CG", 2))
    uex = fem.Function(V2)
    uex.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)

    L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
    error_local = fem.assemble_scalar(L2_error)
    error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

    error_max = np.max(np.abs(uD.x.array - uh.x.array))

    # Only print the error on one process
    if domain.comm.rank == 0:
        print(f"Error_L2 : {error_L2:.2e}")
        print(f"Error_max : {error_max:.2e}")

    u_pred = eval_func(uh, XY_eval, domain)

    return u_pred


N = 32


xsamp = np.linspace(0, 1, N + 1)
X, Y = np.meshgrid(xsamp, xsamp)
print(X.shape)
XYZ = np.stack((X, Y, 0 * Y), axis=0)
print(XYZ.shape)
XYZ_flat = XYZ.reshape(3, -1)
print(XYZ_flat.shape)

u_pred_big = solve_FEA(N, u_true_np, f_driving, XYZ_flat).reshape(N + 1, N + 1)
# u_pred_big = u_pred_big.reshape(N + 1, N + 1)

u_true_samp = u_true_np(XYZ_flat[:2]).reshape(N + 1, N + 1)

print(u_pred_big.shape, u_true_samp.shape)


from matplotlib import pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
# plt.imshow()
im0 = ax[0].imshow(u_true_samp)
fig.colorbar(im0, ax=ax[0])
ax[0].set_title("True field")

im1 = ax[1].imshow(u_pred_big)
fig.colorbar(im1, ax=ax[1])
ax[1].set_title("Predicted field")

im2 = ax[2].imshow(abs(u_pred_big - u_true_samp))
fig.colorbar(im2, ax=ax[2])
ax[2].set_title("Error")
fig.tight_layout()
plt.show()
