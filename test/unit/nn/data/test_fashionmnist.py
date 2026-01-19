import torch
import torchvision.transforms as transforms

from src.nn.data import fashionmnist


class _DummyDataset:
    def __init__(self, storage_path, train, transform, download):
        self.storage_path = storage_path
        self.train = train
        self.transform = transform
        self.download = download
        self.class_to_idx = {"shirt": 0, "shoe": 1}

    def __getitem__(self, index):
        return torch.zeros(1, 28, 28), 0


class _DummyLoader:
    def __init__(self, dataset, shuffle, batch_size, **_kwargs):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __len__(self):
        return 1


def test_fashionmnist_builds_loaders_and_metadata(monkeypatch):
    created_datasets = []
    created_loaders = []

    def fake_dataset(storage_path, train, transform, download):
        dataset = _DummyDataset(storage_path, train, transform, download)
        created_datasets.append(dataset)
        return dataset

    def fake_dataloader(dataset, shuffle, batch_size, **kwargs):
        loader = _DummyLoader(dataset, shuffle, batch_size, **kwargs)
        created_loaders.append(loader)
        return loader

    monkeypatch.setattr(
        fashionmnist.torchvision.datasets, "FashionMNIST", fake_dataset
    )
    monkeypatch.setattr(fashionmnist, "DataLoader", fake_dataloader)

    dataset = fashionmnist.FashionMNIST(storage_path="data-root", batch_size=32)

    assert dataset.name == "FashionMNIST"
    assert dataset.classes == {"shirt": 0, "shoe": 1}
    assert created_datasets[0].train is True
    assert created_datasets[1].train is False
    assert created_datasets[0].download is True
    assert created_datasets[1].download is True
    assert created_datasets[0].transform is created_datasets[1].transform
    assert created_loaders[0].shuffle is True
    assert created_loaders[1].shuffle is False
    assert created_loaders[0].batch_size == 32
    assert created_loaders[1].batch_size == 32

    transform_steps = created_datasets[0].transform.transforms
    assert len(transform_steps) == 3
    assert isinstance(transform_steps[0], transforms.ToTensor)
    assert isinstance(transform_steps[1], transforms.Normalize)
    assert isinstance(transform_steps[2], transforms.Lambda)


def test_fashionmnist_appends_custom_transform(monkeypatch):
    created_datasets = []

    def fake_dataset(storage_path, train, transform, download):
        dataset = _DummyDataset(storage_path, train, transform, download)
        created_datasets.append(dataset)
        return dataset

    monkeypatch.setattr(
        fashionmnist.torchvision.datasets, "FashionMNIST", fake_dataset
    )
    monkeypatch.setattr(fashionmnist, "DataLoader", _DummyLoader)

    extra_transform = transforms.Compose([transforms.Lambda(lambda x: x)])

    fashionmnist.FashionMNIST(transform=extra_transform)

    transform_steps = created_datasets[0].transform.transforms
    assert len(transform_steps) == 4
    assert isinstance(transform_steps[0], transforms.ToTensor)
    assert isinstance(transform_steps[1], transforms.Normalize)
    assert isinstance(transform_steps[2], transforms.Lambda)
    assert transform_steps[3] is extra_transform.transforms[0]
