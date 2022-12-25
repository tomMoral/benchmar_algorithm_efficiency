from abc import abstractmethod
from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import jax
    import jax.random

    from algorithmic_efficiency import random_utils as prng


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class SubmissionSolver(BaseSolver):

    stopping_criterion = SufficientProgressCriterion(
        patience=10, strategy="callback"
    )

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        'batch_size': [1024],
        'rng': [33],
    }

    def skip(self, workload, data_dir, framework):
        if framework == 'jax':
            return True, "Only for framework=='pytorch'"
        return False, None

    def set_objective(self, workload, data_dir, framework):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.workload, self.data_dir = workload, data_dir
        self.framework = framework

        seed = self.rng
        if framework == 'jax':
            seed = jax.random.PRNGKey(self.rng)

        (data_rng, self.opt_init_rng,
         self.model_init_rng, self.run_rng) = prng.split(seed, 4)

        self.input_queue = workload._build_input_queue(
            data_rng,
            'train',
            data_dir=data_dir,
            global_batch_size=self.batch_size
        )

    @staticmethod
    def get_next(stop_val):
        return stop_val + 20

    @abstractmethod
    def init_optimizer_state(self, workload, model_params, model_state, rng):
        pass

    @abstractmethod
    def data_selection(self, optimizer_state, model_params,
                       model_state, global_step, rng):
        """Select data from the infinitely repeating, pre-shuffled input queue.

        Each element of the queue is a batch of training examples and labels.
        """
        pass

    @abstractmethod
    def update_params(self, optimizer_state, model_params, model_state, batch,
                      eval_results, global_step, rng):
        """Return (updated_optimizer_state, updated_params)."""

        pass

    def run(self, cb):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.*

        # Init the model weights
        model_params, model_state = self.workload.init_model_fn(
            self.model_init_rng
        )
        optimizer_state = self.init_optimizer_state(
            self.workload, model_params, model_state, self.opt_init_rng
        )

        global_step = 0
        seed = self.rng
        if self.framework == 'jax':
            seed = jax.random.PRNGKey(seed)

        while cb((model_params, model_state)):
            step_rng = prng.fold_in(seed, global_step)
            data_select_rng, update_rng, eval_rng = prng.split(step_rng, 3)
            batch = self.data_selection(
                optimizer_state, model_params, model_state, global_step,
                data_select_rng
            )

            optimizer_state, model_params, model_state = self.update_params(
                optimizer_state, model_params, model_state, batch,
                eval_results=None, global_step=global_step, rng=update_rng
            )

            global_step += 1

        self.model = (model_params, model_state)

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return self.model


class TorchSubmissionSolver(SubmissionSolver):

    test_config = {
        'dataset': dict(framework='pytorch')
    }

    def skip(self, workload, data_dir, framework):
        if framework != 'pytorch':
            return True, "Only for framework=='pytorch'"
        return False, None


class JaxSubmissionSolver(SubmissionSolver):

    test_config = {
        'dataset': dict(framework='jax')
    }

    def skip(self, workload, data_dir, framework):
        if framework != 'jax':
            return True, "Only for framework=='jax'"
        return False, None
