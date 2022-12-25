from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import algorithmic_efficiency  # noqa: F401
    import jax.random


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "ML Commons Algorithm Efficiency"

    install_cmd = "conda"
    requirements = [
        "pip:git+https://github.com/mlcommons/algorithmic-efficiency"
        "#egg=algorithmic_efficiency[full]"
    ]

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.3"

    parameter = {
        'eval_batch_size': 1024,
    }

    def set_data(self, workload, data_dir, framework):
        self.workload, self.data_dir = workload, data_dir
        self.framework = framework

        self.eval_rng = 27
        self.model_init_rng = 42
        if framework == 'jax':
            self.eval_rng = jax.random.PRNGKey(self.eval_rng)
            self.model_init_rng = jax.random.PRNGKey(self.model_init_rng)

    def compute(self, model):
        model_params, model_state = model

        objective = self.workload.eval_model(
            self.workload.eval_batch_size,
            model_params,
            model_state,
            self.eval_rng,
            self.data_dir,
            None,
            0
        )
        objective['value'] = objective['test/loss']
        return objective

    def get_one_solution(self):
        # Return one solution. The return value should be an object compatible
        # with `self.compute`. This is mainly for testing purposes.
        return self.workload.init_model_fn(self.model_init_rng)

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        return dict(
            workload=self.workload, data_dir=self.data_dir,
            framework=self.framework
        )
