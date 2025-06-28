import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid


class FacialEmotionDataLoader:
    def __init__(
        self,
        dataset_path: str,
        model_type: str,
        image_size: tuple = (48, 48),
        batch_size: int = 32,
        num_workers: int = 4,
        split_ratio: float = 0.8,
        seed: int = 42,
    ) -> None:
        self.dataset_path = dataset_path
        self.model_type = model_type
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio
        self.seed = seed

        self._init_transforms()
        self._load_dataset()

    def _init_transforms(self):
        if self.model_type == "cnn":
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        elif self.model_type == "resnet":
            self.transforms = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            raise ValueError("Invalid model type. Choose 'cnn' or 'resnet'.")

    def _load_dataset(self) -> None:
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"{self.dataset_path} does not exist.")

        self.full_dataset = datasets.ImageFolder(
            root=self.dataset_path, transform=self.transforms
        )
        self.classes = self.full_dataset.classes

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, List[str]]:
        generator = torch.Generator().manual_seed(self.seed)
        train_size = int(self.split_ratio * len(self.full_dataset))
        test_size = len(self.full_dataset) - train_size
        train_dataset, test_dataset = random_split(
            self.full_dataset, [train_size, test_size], generator=generator
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return train_loader, test_loader, self.classes

    @staticmethod
    def imshow(img) -> None:
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis("off")
        plt.show()

    @staticmethod
    def unnormalize(img, model_type):
        if model_type == "cnn":
            mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        elif model_type == "resnet":
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        else:
            raise ValueError("Invalid model type for unnormalization.")

        return img * std + mean


if __name__ == "__main__":
    dataset_path = "./data"
    model_types = ["cnn", "resnet"]

    for model_type in model_types:
        loader = FacialEmotionDataLoader(
            dataset_path=dataset_path, model_type=model_type
        )

        train_loader, test_loader, class_names = loader.get_data_loaders()

        print(f"\nDetected Emotion Classes: {class_names}")
        print(f"Total Training Samples: {len(train_loader.dataset)}")
        print(f"Total Testing Samples: {len(test_loader.dataset)}")
        print(f"Batch Size: {loader.batch_size}")
        print(f"Image Size: {loader.image_size}")

        data_iter = iter(train_loader)
        images, labels = next(data_iter)

        print(f"\nExample Batch - Image Shape: {images.shape}")
        print(f"Example Batch - Labels Shape: {labels.shape}")

        unnormalized_images = FacialEmotionDataLoader.unnormalize(
            make_grid(images), model_type=model_type
        )
        FacialEmotionDataLoader.imshow(unnormalized_images)
        print(
            "Labels in batch:", " ".join(f"{class_names[labels[j]]}" for j in range(32))
        )
