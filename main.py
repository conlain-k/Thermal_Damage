import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from jax import jit, vmap, jacrev


import matplotlib.pyplot as plt

from functools import partial

# how many samples per batch?
batch_size = 256
input_size = 2
learning_rate = 5e-2
seed = 8

num_epochs = 5000

xmin = -2
xmax = 2

lam = 2

Nprint = 256 + 1

# ignore parameters
def U_true_wrapper(params, x):
    return U_true(x)


@jit
def U_true(xvec):
    x, y = xvec[..., 0], xvec[..., 1]

    xlo = (x <= 0) * 1.0
    xhi = (x > 0) * 1.0
    return 0 * x * y  # zero-Dirichlet
    # return (8 - x**2 - y**2) * xlo + (
    #     4 + 4 * jnp.exp(-x * lam) - y**2
    # ) * xhi  # nonlinear
    # return 1 + 2 * x + 4 * y  # linear
    # return 1 + 2 * x + 4 * y + 6 * x * y  # bilinear
    # return 1 + 2 * x + 4 * y + 6 * x * y + 8 * x**2 + 10 * y**2  # quadratic


@jit
def k(xvec):
    x, y = xvec[..., 0], xvec[..., 1]
    xlo = (x <= 0) * 1.0
    xhi = (x > 0) * 1.0

    return xlo + 5 * xhi


@jit
def H_true(xvec):
    x, y = xvec[..., 0], xvec[..., 1]

    xlo = (x <= 0) * 1.0
    xhi = (x > 0) * 1.0

    lapl = 4 * xlo + (2 + 9 * jnp.cos(x)) * xhi

    lap = lapl_op(U_true)(xvec)

    # if U_True is prescribed
    Hx = -lap * k(xvec)

    # if we want to prescribe H instead
    return jnp.exp(-(x**2 + y**2) / 1)

    # return


def MAE(x1, x2):
    return abs(x1 - x2).sum()


def MSE(x1, x2):
    return ((x1 - x2) ** 2).sum()


def loss(x1, x2):
    return MSE(x1, x2)


class PINN(nn.Module):
    @nn.compact
    def __call__(self, x):
        # make sure we have a batch dimension
        if len(x.shape) == 0:
            x = x.reshape(1, -1)
            print("XXX", x.shape)
            exit(-1)
        # note that here x represents the input vector = [x_i, t_i]

        # comment out pairs of lines to add more hidden layers
        # # 2 HL
        x = nn.Dense(features=64)(x)
        x = nn.softplus(x)

        # 1HL
        x = nn.Dense(features=64)(x)
        x = nn.softplus(x)

        x = nn.Dense(features=1)(x)

        return x


# def grad_op(f):
#     # take gradient of single-instance function and map it over domain
#     return jax.vmap(jax.jacfwd(f))


def egrad_op(g):
    # taken from JAX forums: https://github.com/google/jax/issues/3556
    def wrapped(x, *rest):
        y, g_vjp = jax.vjp(lambda x: g(x, *rest), x)
        (x_bar,) = g_vjp(jnp.ones_like(y))
        return x_bar

    return wrapped


def div_op(g):
    # laplacian without sum
    div_unsum = egrad_op(g)

    def div_g(x):
        # bind x in as an argument, sum over last axis (assumes non-scalar input)
        return div_unsum(x).sum(axis=-1)

    # return function operator
    return div_g


def lapl_op(g):
    # get diagonal terms from the hessian (for each instance)
    hess_g = jax.hessian(g)

    def lapl_g(x):
        return jnp.diag(hess_g(x).squeeze()).sum(axis=-1)

    # return laplacian for each instance
    return lapl_g


# given sensors x, compute the loss for model u given params
@partial(jit, static_argnames="u")
@partial(jax.vmap, in_axes=(None, None, 0))
def PINN_loss(u, params, batch):
    # split inputs and outputs
    (X, y) = batch

    # split locations from times
    # (xvec, tvec) = X
    # wrapper function for binding current parameters and t onto network
    def u_x(xv):
        # concat x and t vectors along second axis (channel-wise)
        # xt = jnp.concatenate((xv, tvec), axis=1)
        return u({"params": params}, xv)

    # # bind in params and t now
    # def u_t(tv):
    #     # concat x and t vectors along second axis (channel-wise)
    #     xt = jnp.concatenate((xvec, tv), axis=1)
    #     return u({"params": params}, xt)

    # now evaluate each term on given parameters at given sensors
    # uv = u_x(X)
    lapl_u_x = lapl_op(u_x)(X)
    # grad_u_t = grad_op(u_t)(X)

    # get heat eqn resid
    resid = k(X) * lapl_u_x + H_true(X)  # + grad_u_t

    # print(k(X) * lapl_u_x)
    # print(H_true(X))

    # PINN loss is squared PDE residual, summed across input
    PDE_loss = loss(resid, jnp.zeros_like(resid))

    return PDE_loss


@partial(jit, static_argnames="u")
@partial(jax.vmap, in_axes=(None, None, 0))
def supervised_loss(u, params, batch):
    X, y = batch
    # apply model to given x vals
    y_pred = u({"params": params}, X)

    # return mean squared error between target and predicted
    return loss(y, y_pred)


@partial(jit, static_argnames="u")
@partial(jax.vmap, in_axes=(None, None, 0))
def BC_loss(u, params, bcs_batch):
    bc_X, bc_y = bcs_batch
    u_bcs = u({"params": params}, bc_X)

    bc_loss = loss(u_bcs, bc_y)

    return bc_loss


@jit
def train_step(state, batch, bcs_batch):

    (x, y) = batch

    # bind all loss functions into one joint term (map in model u and sensors x)
    def combined_loss(params):
        # comput loss for each instance in batch, then sum
        # sup = 0.0 * supervised_loss(state.apply_fn, params, batch).mean()
        pde = PINN_loss(state.apply_fn, params, batch).mean()
        bcs = BC_loss(state.apply_fn, params, bcs_batch).mean()

        # scaled_loss = (sup**2 + pde**2 + bcs**2) / (sup + pde + bcs)

        sum_loss = pde + bcs

        # compute scaled MSE
        loss = sum_loss

        # RMSE
        loss = jnp.sqrt(loss)

        # compute combined loss, but return them individually
        return loss, {"pde": pde, "bcs": bcs}

    # get gradients for current parameter assignment
    (_, metrics), grads = jax.value_and_grad(combined_loss, has_aux=True)(state.params)
    # loss, grads = jax.value_and_grad(supervised_loss)(state.params)

    # update weights using optimizer
    return metrics, state.apply_gradients(grads=grads), grads


def make_state(rng):
    model = PINN()
    sample_input = jnp.ones(shape=(input_size))
    variables = model.init(rng, sample_input)  # Initialization call

    # construct lr sched
    schedule = optax.cosine_onecycle_schedule(
        peak_value=learning_rate,
        transition_steps=num_epochs,
    )
    # construct optimizer to use lr sched
    optimizer = optax.adamw(learning_rate=schedule, weight_decay=1e-5)

    # now build full training state
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=variables["params"], tx=optimizer
    )

    return state


# sample x given rng, then update the rng seed
def sample_x(count, rng):
    rng, _ = jax.random.split(rng)
    x_i = jax.random.uniform(rng, shape=(count, input_size), minval=xmin, maxval=xmax)
    return x_i, rng


def sample_bcs(count, rng):
    def sample_1d(c, rng):
        # now update the rng seed
        rng, _ = jax.random.split(rng)
        samp_vals = jax.random.uniform(
            rng, shape=(c, input_size), minval=xmin, maxval=xmax
        )

        return samp_vals, rng

    xlo, rng = sample_1d(count // 4, rng)
    xhi, rng = sample_1d(count // 4, rng)
    ylo, rng = sample_1d(count // 4, rng)
    yhi, rng = sample_1d(count // 4, rng)
    # also get min and max vals
    mins, maxes = (
        jnp.ones(shape=(count // 4, 1)) * xmin,
        jnp.ones(shape=(count // 4, 1)) * xmax,
    )

    # override these terms to be external
    # jax makes this kinda janky lol
    xlo = xlo.at[:, [1]].set(mins)
    xhi = xhi.at[:, [1]].set(maxes)
    ylo = ylo.at[:, [0]].set(mins)
    yhi = yhi.at[:, [0]].set(maxes)

    # now batch all the bc points up at once
    return jnp.concatenate((xlo, xhi, ylo, yhi), axis=0), rng


if __name__ == "__main__":

    rng = jax.random.PRNGKey(0)  # PRNG Key
    # build optimizer state
    state = make_state(rng)

    rng = jax.random.PRNGKey(seed)
    for e in range(num_epochs):

        x_i, rng = sample_x(batch_size, rng)
        x_bcs, rng = sample_bcs(batch_size, rng)
        # print(x_bcs)
        u_i = U_true(x_i)
        u_bcs = U_true(x_bcs)

        metrics, state, grads = train_step(state, (x_i, u_i), (x_bcs, u_bcs))

        if e % (num_epochs // 20) == 0:
            print(f"Epoch {e}: losses are {metrics}")

    u_batched = vmap(state.apply_fn, (None, 0))

    x_bcs_test, rng = sample_bcs(1000, rng)
    U_bcs = U_true(x_bcs_test)
    X, Y = jnp.meshgrid(
        jnp.linspace(xmin, xmax, Nprint), jnp.linspace(xmin, xmax, Nprint)
    )
    XY = jnp.stack((X.ravel(), Y.ravel()), axis=-1)
    print(XY.shape)

    # print(u_batched({"params": state.params}, endpoints))
    # print(U_true(endpoints))

    # print("True solution PDE loss", PINN_loss(U_true_wrapper, None, (x_i_test, None)))
    # print("Approx soln PDE loss", PINN_loss(state.apply_fn, state.params, (x_i_test, None)))

    print("True solution PDE loss", PINN_loss(U_true_wrapper, None, (XY, None)).mean())
    print(
        "True solution BC loss",
        BC_loss(U_true_wrapper, None, (x_bcs_test, U_bcs)).mean(),
    )
    print(
        "Approx soln PDE loss",
        PINN_loss(state.apply_fn, state.params, (XY, None)).mean(),
    )
    print(
        "Approx soln BC loss",
        BC_loss(state.apply_fn, state.params, (x_bcs_test, U_bcs)).mean(),
    )

    u_pred = u_batched({"params": state.params}, XY)
    u_pred = u_pred.reshape(Nprint, Nprint)
    print(u_pred.shape)
    u_true = U_true(XY).reshape(Nprint, Nprint)

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    # plt.imshow()
    im0 = ax[0, 0].imshow(u_true)
    fig.colorbar(im0, ax=ax[0, 0])
    ax[0, 0].set_title("True field")

    im1 = ax[0, 1].imshow(u_pred)
    fig.colorbar(im1, ax=ax[0, 1])
    ax[0, 1].set_title("Predicted field")

    im2 = ax[0, 2].imshow(abs(u_true - u_pred))
    fig.colorbar(im2, ax=ax[0, 2])
    ax[0, 2].set_title("Error")
    # fig.tight_layout()
    # plt.savefig("PINN_results.png", dpi=300)

    kplot = jax.vmap(k)(XY).reshape(Nprint, Nprint)
    Hplot = jax.vmap(H_true)(XY).reshape(Nprint, Nprint)
    pde_resid = PINN_loss(state.apply_fn, state.params, (XY, None)).reshape(
        Nprint, Nprint
    )

    # get x vals along y = 50
    XY_r = XY.reshape(Nprint, Nprint, 2)
    xplot = XY_r[50, :, 0]
    # print(XY_r[50, :])
    ax[1, 0].plot(xplot, u_true[50, :], label="True")
    ax[1, 0].plot(xplot, u_pred[50, :], label="Predicted")
    ax[1, 0].legend()
    ax[1, 0].set_title("Slice of solutions along $y=0$")

    im1 = ax[1, 1].imshow(Hplot)
    fig.colorbar(im1, ax=ax[1, 1])
    ax[1, 1].set_title("$H(x)$")

    fig.colorbar(im2, ax=ax[1, 2])
    ax[1, 2].set_title("PDE Residual")
    im0 = ax[1, 2].imshow(pde_resid)

    fig.tight_layout()
    plt.savefig("PINN_results.png", dpi=300)

    # make smaller plots intended for last case study
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    im0 = ax[0].imshow(u_pred)
    fig.colorbar(im0, ax=ax[0])
    ax[0].set_title("PINN Predicted Solution")

    im1 = ax[1].imshow(pde_resid)
    fig.colorbar(im1, ax=ax[1])
    ax[1].set_title("PDE Residual")

    fig.tight_layout()
    plt.savefig("PINN_small.png", dpi=300)

    plt.show()
