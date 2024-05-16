"""A Task for evaluating the classification accuracy on ImageNet-R."""

import dataclasses
import os

import numpy as np
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms

import shifthappens.data.base as sh_data
import shifthappens.data.torch as sh_data_torch
import shifthappens.utils as sh_utils
from shifthappens import benchmark as sh_benchmark
from shifthappens.data.base import DataLoader
from shifthappens.models import base as sh_models
from shifthappens.models.base import PredictionTargets
from shifthappens.tasks.base import Task
from shifthappens.tasks.metrics import Metric
from shifthappens.tasks.task_result import TaskResult


@sh_benchmark.register_task(
    name="ImageNet-A", relative_data_folder="imagenet_a", standalone=True
)
@dataclasses.dataclass
class ImageNetA(Task):
    """Measures the classification accuracy on ImageNet-A [1], a dataset
    containing natural adversarial examples.

    [1] Natural Adversarial Examples. Dan Hendrycks and Kevin Zhao and Steven Basart and Jacob
    Steinhardt and Dawn Son. CVPR. 2021
    """

    resources = [
        (
            "imagenet-a.tar",
            "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar",
            "c3e55429088dc681f30d81f4726b6595",
        )
    ]

    def setup(self):
        """Load and prepare the data."""

        dataset_folder = os.path.join(self.data_root, "imagenet-a")
        if not os.path.exists(dataset_folder):
            # download data
            for file_name, url, md5 in self.resources:
                sh_utils.download_and_extract_archive(
                    url, self.data_root, md5, file_name
                )

        test_transform = tv_transforms.Compose(
            [
                tv_transforms.ToTensor(),
                tv_transforms.Lambda(lambda x: x.permute(1, 2, 0)),
            ]
        )

        self.ch_dataset = tv_datasets.ImageFolder(
            root=dataset_folder, transform=test_transform
        )
        self.images_only_dataset = sh_data_torch.IndexedTorchDataset(
            sh_data_torch.ImagesOnlyTorchDataset(self.ch_dataset)
        )

    def _prepare_dataloader(self) -> DataLoader:
        return sh_data.DataLoader(self.images_only_dataset, max_batch_size=None)

    def _evaluate(self, model: sh_models.Model) -> TaskResult:
        dataloader = self._prepare_dataloader()

        all_predicted_labels_list = []
        for predictions in model.predict(
            dataloader, PredictionTargets(class_labels=True)
        ):
            all_predicted_labels_list.append(predictions.class_labels)
        all_predicted_labels = np.concatenate(all_predicted_labels_list, 0)

        accuracy = np.equal(
            all_predicted_labels, np.array(self.ch_dataset.targets)
        ).mean()

        return TaskResult(
            accuracy=accuracy, summary_metrics={Metric.Robustness: "accuracy"}
        )
