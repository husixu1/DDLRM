import alpa
from alpa.testing import assert_allclose
import copy
from flax import linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from jax import random
import optax
import ray

# INIT ########################################################################
alpa.util.disable_tqdm_globally()

ray.init()
alpa.init(cluster="ray")


# MODEL #######################################################################
class MLPModel(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim * 4)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim * 4)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        return x


dim = 2048
batch_size = 2048

# Generate ground truth W and b
rngkey = jax.random.PRNGKey(0)
k1, k2 = random.split(rngkey)
W = random.normal(k1, (dim, dim))
b = random.normal(k2, (dim,))

# Generate the training data
ksample, knoise = random.split(k1)
x = random.normal(ksample, (batch_size, dim))
y = (x @ W + b) + 0.1 * random.normal(knoise, (batch_size, dim))

# Initialize a train state, which includes the model paramter and optimizer
# state.
model = MLPModel(hidden_dim=dim)
params = model.init(rngkey, x)
tx = optax.adam(learning_rate=1e-3)
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# Define the training step
def train_step(state, batch):

    def loss_func(params):
        out = model.apply(params, batch["x"])
        loss = jnp.mean((out - batch["y"])**2)
        return loss

    grads = jax.grad(loss_func)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state


batch = {"x": x, "y": y}
expected_state = train_step(state, batch)


# MANUAL_PIPELINE #############################################################
# Define a MLP model with manual stage boundaries.
class ManualPipelineMLPModel(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim * 4)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        # Use this boundary marker to separate the model into two stages.
        alpa.mark_pipeline_boundary()
        x = nn.Dense(features=self.hidden_dim * 4)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        return x


# Initialize the train state with the same parameters as the single-device
# model.
manual_pipeline_model = ManualPipelineMLPModel(hidden_dim=dim)
manual_pipeline_state = TrainState.create(apply_fn=manual_pipeline_model.apply,
                                          params=copy.deepcopy(params),
                                          tx=tx)


# Define the training step.
# We use the "alpa.PipeshardParallel" option to let alpa use both
# pipeline parallelism and shard parallelism. To make pipeline parallelism
# efficient, we need to fill the pipeline with many micro batches,
# so a `num_micro_batches` should be specified.
@alpa.parallelize(method=alpa.PipeshardParallel(num_micro_batches=16,
                                                layer_option="manual"))
def manual_pipeline_train_step(state, batch):

    def loss_func(params):
        out = state.apply_fn(params, batch["x"])
        loss = jnp.mean((out - batch["y"])**2)
        return loss

    # We use `alpa.grad` here to separate the apply gradient stage with the
    # forward/backward stages in the pipeline. This is necessary to ensure that
    # the gradient accumulation is correct.
    grads = alpa.grad(loss_func)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state


manual_pipeline_actual_state = manual_pipeline_train_step(
    manual_pipeline_state, batch)
assert_allclose(expected_state.params,
                manual_pipeline_actual_state.params,
                atol=5e-3)

alpa.shutdown()
