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
batch_size = 64
input_size = 1
learning_rate = 1e-3
seed = 7

num_epochs = 4000

xmin = -np.pi
xmax = 2 * np.pi


# ignore parameters
def U_true_wrapper(params, x):
    return U_true(x)


@jit
def U_true(x):
    return x * jnp.cos(2 * x)


class PINN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=64)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=64)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=64)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=1)(x)
        return x


# given sensors x, compute the loss for model u given params
@partial(jit, static_argnames="u")
def PINN_loss(u, params, batch):
    (x, y) = batch
    # wrapper function for binding current parameters onto network
    def _u(x):
        return u({"params": params}, x)

    x, _ = batch

    # get u vals
    u_i = vmap(_u)(x)
    # since u is a scalar output, this should have the same shape (but the jacobian adds some extra singleton dimensions)
    uxx_i = vmap(jacrev(jacrev(_u)))(x)
    print(uxx_i.shape, u_i.shape, x.shape)
    uxx_i = uxx_i.reshape(u_i.shape)

    resid = uxx_i + 4 * (jnp.sin(2 * x) + u_i)
    # PINN loss is squared PDE residual, summed across input
    PDE_loss = (resid**2).mean()

    return PDE_loss


@partial(jit, static_argnames="u")
def supervised_loss(u, params, batch):
    x, y = batch
    # apply model to given x vals
    y_pred = u({"params": params}, x)

    # return mean squared error between target and predicted
    return ((y - y_pred) ** 2).mean()


@partial(jit, static_argnames="u")
def BC_loss(u, params, _):
    endpoints = jnp.array([xmin, xmax]).reshape(2, 1)
    u_end = u({"params": params}, endpoints)

    u_end_true = U_true(endpoints)

    bc_loss = ((u_end - u_end_true) ** 2).mean()

    return bc_loss


@jit
def train_step(state, batch):

    (x, y) = batch

    # bind all loss functions into one joint term (map in model u and sensors x)
    def combined_loss(params):
        sup = supervised_loss(state.apply_fn, params, batch)
        pde = PINN_loss(state.apply_fn, params, batch)
        bcs = BC_loss(state.apply_fn, params, batch)

        scaled_loss = (sup**2 + pde**2 + bcs**2) / (sup + pde + bcs)

        sum_loss = sup + pde + bcs

        # compute combined loss, but return them individually
        return sum_loss, {"sup": sup, "pde": pde, "bcs": bcs}

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
    optimizer = optax.adam(learning_rate=schedule)

    # now build full training state
    state = train_state.TrainState.create(apply_fn=model.apply, params=variables["params"], tx=optimizer)

    return state


if __name__ == "__main__":

    rng = jax.random.PRNGKey(0)  # PRNG Key
    # build optimizer state
    state = make_state(rng)

    rng = jax.random.PRNGKey(seed)
    for e in range(num_epochs):
        rng, _ = jax.random.split(rng)
        x_i = jax.random.uniform(rng, shape=(batch_size, 1), minval=xmin, maxval=xmax)
        u_i = U_true(x_i)

        metrics, state, grads = train_step(state, (x_i, u_i))

        # print(grads)

        if e % (num_epochs // 20) == 0:
            print(f"Epoch {e}: losses are {metrics}")

    endpoints = jnp.array([xmin, xmax]).reshape(2, 1)
    print(state.apply_fn({"params": state.params}, endpoints))
    print(U_true(endpoints))

    x_test = jnp.linspace(xmin, xmax, 500).reshape(-1, 1)

    print("True solution PDE loss", PINN_loss(U_true_wrapper, None, (x_test, None)))
    print("Approx soln PDE loss", PINN_loss(state.apply_fn, state.params, (x_test, None)))

    plt.plot(x_test, state.apply_fn({"params": state.params}, x_test), ".", label="Predicted")
    plt.plot(x_test, U_true(x_test), label="Target")
    plt.legend()
    plt.show()
