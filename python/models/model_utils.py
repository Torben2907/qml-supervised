import jax
from jax import numpy as jnp
import optax


def train(model, loss, optim: optax.optim, X, y, random_key, convergence_interval):
    params = model.params_

    opt = optim(learning_rate=model.learning_rate)
    opt_state = opt.init(params)
    grad_fn = jax.grad(loss)

    if model.jit:
        grad_fn = jax.jit(grad_fn)
