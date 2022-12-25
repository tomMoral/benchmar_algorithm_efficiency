from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import importlib
    from absl import flags

    flags.DEFINE_enum(
        'framework',
        None,
        enum_values=['jax', 'pytorch'],
        help='Whether to use Jax or Pytorch for the submission. Controls '
        'among other things if the Jax or Numpy RNG library is used for RNG.'
    )
    FLAGS = flags.FLAGS


WORKLOAD_MODULE, WORKLOAD_CLASS = (
    "algorithmic_efficiency.workloads.mnist.mnist_{framework}.workload",
    "MnistWorkload"
)


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'framework': ['pytorch', 'jax'],
        'data_dir': ["data"],
        'random_state': [27],
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Generate pseudorandom data using `numpy`.
        workload_module = importlib.import_module(
            WORKLOAD_MODULE.format(framework=self.framework)
        )
        workload_class = getattr(workload_module, WORKLOAD_CLASS)
        # workload = workload_class()

        FLAGS(['benchopt', f"--framework={self.framework}"])

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(
            workload=workload_class(), data_dir=self.data_dir,
            framework=self.framework
        )
