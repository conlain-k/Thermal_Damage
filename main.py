import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

# how many samples per batch?
batch_size = 32
input_size = 1
learning_rate = 1e-4
seed = 7000


@jax.jit
def U_true(x):
    return jnp.sin(x)


class PINN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=32)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=32)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=1)(x)
        return x


@jax.jit
def train_step(state, batch):
    # compute loss for set of weights
    def loss_fn(params):
        x, y = batch
        # apply model to given x vals
        y_pred = state.apply_fn({"params": params}, x)

        # return mean squared error between target and predicted
        return ((y - y_pred) ** 2).mean()

    # get gradients for current parameter assignment
    loss, grads = jax.value_and_grad(loss_fn(state.params))

    print(loss)

    # update weights using optimizer
    return state.apply_gradients(grads)


# def main():
#     key = jax.random.PRNGKey(seed)
#     for e in range(100):
#         key, subkey = jax.random.split(key)
#     pass


# # make a loss function for a set of parameters, to be evaluated on x
# def make_loss_fn(params):
#     # get a function wrapper for u
#     def u_fn(params):
#         def _u(x):
#             return model.apply({"params": params}, x)

#         return jax.jit(_u)

#     # now make a pinn loss function, given params

#     @jax.jit
#     def pinn_loss(x):

#         u = u_fn(params)(x).squeeze()
#         uxx = jax.jacrev(jax.jacrev(u_fn(params)))(x).squeeze()

#     return u_fn(params)(x) + uxx_fn(params)(x)


def make_state(rng):
    model = PINN()
    sample_input = jnp.ones(shape=(input_size))
    variables = model.init(rng, sample_input)  # Initialization call
    params = variables["params"]
    optimizer = optax.adam(learning_rate)
    # construct optimizer state
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )

    return state


if __name__ == "__main__":

    rng = jax.random.PRNGKey(0)  # PRNG Key
    # build optimizer state
    state = make_state(rng)

    rng = jax.random.PRNGKey(seed)
    for e in range(100):
        rng, _ = jax.random.split(rng)
        x_i = jax.random.uniform(rng).reshape(-1, 1)
        u_i = U_true(x_i)

        state = train_step(state, (x_i, u_i))

    x_i = jnp.linspace(0, 2 * np.pi, 500)
    import matplotlib.plt as pyplot

    plt.plot(x, state.apply_fn({"params": state.params}, x))
    plt.show()
