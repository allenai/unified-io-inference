from functools import partial
from jax import grad, lax
import jax.numpy as jnp
import jax as jax
print('<<< jax test >>>')
print(jax.devices())

def tanh(x):  # Define a function
  y = jnp.exp(-2.0 * x)
  return (1.0 - y) / (1.0 + y)

grad_tanh = grad(tanh)  # Obtain its gradient function
print(grad_tanh(1.0)) 
print('<<< end >>>')
