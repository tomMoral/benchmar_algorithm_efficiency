from benchopt import safe_import_context

import functools
from benchmark_utils.submission_solver import JaxSubmissionSolver

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:

    from flax import jax_utils
    import jax
    from jax import lax
    import jax.numpy as jnp
    import optax

    from algorithmic_efficiency import spec


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(JaxSubmissionSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'Adam-jax'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        'lr': [1e-3],
        'epsilon': [1e-5],
        'beta_1': [1e-1, 1e-2],
        **JaxSubmissionSolver.parameters
    }

    def init_optimizer_state(self, workload, model_params, model_state, rng):
        del model_params
        del model_state
        del rng
        params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple),
                                         workload.param_shapes)
        opt_init_fn, opt_update_fn = optax.chain(
            optax.scale_by_adam(
                b1=self.beta_1,
                b2=0.999,
                eps=self.epsilon),
            optax.scale(-self.lr)
        )
        return (
            jax_utils.replicate(opt_init_fn(params_zeros_like)),
            opt_update_fn
        )

    def data_selection(self, optimizer_state, model_params,
                       model_state, global_step, rng):
        """Select data from the infinitely repeating, pre-shuffled input queue.

        Each element of the queue is a batch of training examples and labels.
        """
        return next(self.input_queue)

    def update_params(self, optimizer_state, model_params, model_state, batch,
                      eval_results, global_step, rng):
        """Returns (new_optimizer_state, new_params, new_model_state)."""
        del eval_results
        del global_step

        per_device_rngs = jax.random.split(rng, jax.local_device_count())
        optimizer_state, opt_update_fn = optimizer_state
        new_optimizer_state, updated_params, new_model_state = pmapped_update_params(
            self.workload,
            opt_update_fn,
            model_params,
            model_state,
            None,
            batch,
            optimizer_state,
            per_device_rngs
        )
        return (new_optimizer_state, opt_update_fn), updated_params, new_model_state


# We need to jax.pmap here instead of inside update_params because the latter
# would recompile the function every step.
@functools.partial(
    jax.pmap,
    axis_name='batch',
    in_axes=(None, None, 0, 0, None, 0, 0, 0),
    static_broadcasted_argnums=(0, 1)
)
def pmapped_update_params(workload, opt_update_fn, model_params,
                          model_state, hyperparameters, batch,
                          optimizer_state, rng):
    del hyperparameters

    def loss_fn(params):
        logits_batch, new_model_state = workload.model_fn(
            params,
            batch,
            model_state,
            spec.ForwardPassMode.TRAIN,
            rng,
            update_batch_norm=True)
        loss, _ = workload.loss_fn(batch['targets'], logits_batch)
        return loss, new_model_state

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, new_model_state), grad = grad_fn(model_params)
    grad = lax.pmean(grad, axis_name='batch')
    updates, new_optimizer_state = opt_update_fn(grad, optimizer_state,
                                                 model_params)
    updated_params = optax.apply_updates(model_params, updates)
    return new_optimizer_state, updated_params, new_model_state