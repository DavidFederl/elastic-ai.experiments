import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .dataset import Dataset


class FashionMNIST(Dataset):
    def __init__(
        self,
        storage_path: str = "data",
        transform: transforms.Compose | None = None,
        batch_size: int = 512,
    ) -> None:
        """Instantiate training and validation data loaders for the Fashion-MNIST dataset.

        INFO:
            Elements are transformed to tensors and normalized to mean 0.5 and std 0.5.
            Element size is 1x28x28 (grayscale images of 28x28 pixels).

        Args:
            storage_path (str): Path to store/load the dataset. (Default: "data")
            transform (transforms.Compose | None): Transformations to apply to the data.
                                                   (Default: None)
            seed(int|None): seed for manual seeding! (Default: None)

        Returns:
            None
        """
        transformation_steps: list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
        if transform is not None:
            transformation_steps.extend(list(transform.transforms))
        transformation: transforms.Compose = transforms.Compose(transformation_steps)

        loader_kwargs: dict[str, object] = {
            "batch_size": batch_size,
        }

        training_set = torchvision.datasets.FashionMNIST(
            storage_path, train=True, transform=transformation, download=True
        )
        self.training_loader = DataLoader(
            training_set,
            shuffle=True,
            **loader_kwargs,  # type: ignore
        )

        validation_set = torchvision.datasets.FashionMNIST(
            storage_path, train=False, transform=transformation, download=True
        )
        self.validation_loader = DataLoader(
            validation_set,
            shuffle=False,
            **loader_kwargs,  # type: ignore
        )

        self.name = "FashionMNIST"
        self.classes = training_set.class_to_idx

        super().__init__()
