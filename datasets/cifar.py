from benchmark_utils.workload_dataset import WorkloadDataset


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(WorkloadDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Cifar"

    workload_module = (
        "algorithmic_efficiency.workloads.cifar.cifar_{framework}.workload"
    )
    workload_class = "CifarWorkload"
