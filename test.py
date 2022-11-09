import jax
import jax.numpy as jnp


def egrad_op(g):
    # taken from JAX forums: https://github.com/google/jax/issues/3556
    def wrapped(x, *rest):
        y, g_vjp = jax.vjp(lambda x: g(x, *rest), x)
        (x_bar,) = g_vjp(jnp.ones_like(y))
        return x_bar

    return wrapped


def lapl_op(g):
    # get diagonal terms from the hessian (for each instance)
    hess_g = jax.hessian(g)

    def lapl_g(x):
        # print("here", hess_g(x))
        return jnp.diag(hess_g(x)).sum(axis=-1)

    # return laplacian for each instance
    return jax.vmap(lapl_g)


key = jax.random.PRNGKey(0)

a = jax.random.normal(key, (2, 3))

A = jax.random.uniform(key, (3, 3))
A = (A + A.T) / 2


def f_(x):
    print("x", x.shape)
    return jnp.matmul(jnp.matmul(x.T, A), x)


f = jax.vmap(f_, in_axes=(0))


print(a, A, f(a))

1 / 0
