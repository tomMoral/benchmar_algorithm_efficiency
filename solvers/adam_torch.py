from benchopt import safe_import_context

from benchmark_utils.submission_solver import TorchSubmissionSolver

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import torch
    from algorithmic_efficiency import spec


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(TorchSubmissionSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'Adam-torch'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        'lr': [1e-4, 1e-3],
        **TorchSubmissionSolver.parameters,
    }

    def init_optimizer_state(self, workload, model_params, model_state, rng):
        del rng
        del model_state
        del workload

        optimizer_state = {
            'optimizer': torch.optim.Adam(
                model_params.parameters(), lr=self.lr
            )
        }
        return optimizer_state

    def data_selection(self, optimizer_state, model_params,
                       model_state, global_step, rng):
        """Select data from the infinitely repeating, pre-shuffled input queue.

        Each element of the queue is a batch of training examples and labels.
        """
        return next(self.input_queue)

    def update_params(self, optimizer_state, model_params, model_state, batch,
                      eval_results, global_step, rng):
        """Return (updated_optimizer_state, updated_params)."""

        current_model, optimizer = model_params, optimizer_state['optimizer']

        # Use training mode and zero out the gradient
        current_model.train()
        optimizer.zero_grad()

        # Forward pass
        output, new_model_state = self.workload.model_fn(
            params=current_model,
            augmented_and_preprocessed_input_batch=batch,
            model_state=model_state,
            mode=spec.ForwardPassMode.TRAIN,
            rng=rng,
            update_batch_norm=True
        )
        loss, _ = self.workload.loss_fn(
            label_batch=batch['targets'], logits_batch=output
        )

        # Backward pass and parameter update
        loss.backward()
        optimizer.step()

        return (optimizer_state, model_params, new_model_state)
