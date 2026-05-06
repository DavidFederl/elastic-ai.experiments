from unittest.mock import MagicMock, patch

from torchvision import transforms

from src.nn.data.fashionmnist import (
    FashionMNIST,
    fashionmnist_trainingset_flattened,
    fashionmnist_validationset_flattened,
)


def test_fashionmnist_class_structure():
    """Test that FashionMNIST class has the correct structure."""
    # Test that it inherits from torchvision.datasets.FashionMNIST
    import torchvision.datasets

    assert issubclass(FashionMNIST, torchvision.datasets.FashionMNIST)


def test_fashionmnist_flattened_functions():
    """Test the convenience functions for flattened datasets."""
    with patch("src.nn.data.fashionmnist.FashionMNIST") as mock_fashionmnist:
        mock_instance = MagicMock()
        mock_fashionmnist.return_value = mock_instance

        # Test training set function
        _ = fashionmnist_trainingset_flattened()
        # Check that FashionMNIST was called with correct parameters
        assert mock_fashionmnist.call_count == 1
        call_args = mock_fashionmnist.call_args
        assert call_args[1]["root"] == "datasets"
        assert call_args[1]["train"]
        assert isinstance(call_args[1]["transform"], transforms.Compose)

        # Verify the transform pipeline
        transform = call_args[1]["transform"]
        transforms_list = list(transform.transforms)
        assert len(transforms_list) == 3
        assert isinstance(transforms_list[0], transforms.ToTensor)
        assert isinstance(transforms_list[1], transforms.Normalize)
        assert isinstance(transforms_list[2], transforms.Lambda)

        # Test validation set function
        _ = fashionmnist_validationset_flattened()
        assert mock_fashionmnist.call_count == 2
        call_args = mock_fashionmnist.call_args
        assert call_args[1]["root"] == "datasets"
        assert not call_args[1]["train"]
        assert isinstance(call_args[1]["transform"], transforms.Compose)


def test_fashionmnist_transform_pipeline():
    """Test that the transform pipeline is correctly configured."""
    # Test the training set transform
    train_transform = fashionmnist_trainingset_flattened().transform
    assert isinstance(train_transform, transforms.Compose)

    # Test the validation set transform
    val_transform = fashionmnist_validationset_flattened().transform
    assert isinstance(val_transform, transforms.Compose)

    # Both should have the same transform pipeline
    assert len(list(train_transform.transforms)) == len(list(val_transform.transforms))


def test_fashionmnist_expected_classes():
    """Test that FashionMNIST has the expected class names."""
    # These are the standard FashionMNIST classes
    expected_classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    # Verify we can import and the class structure is correct
    assert hasattr(FashionMNIST, "__init__")
    assert hasattr(FashionMNIST, "classes")  # This will be set by parent class
    assert FashionMNIST.classes == expected_classes


def test_fashionmnist_constructor_signature():
    """Test that FashionMNIST constructor has the expected signature."""
    import inspect

    # Get the constructor signature
    sig = inspect.signature(FashionMNIST.__init__)
    params = list(sig.parameters.keys())

    # Should have the expected parameters
    assert "self" in params
    assert "root" in params
    assert "train" in params
    assert "transform" in params
    assert "target_transform" in params

    # train parameter should default to True
    assert sig.parameters["train"].default
    assert sig.parameters["transform"].default is None
    assert sig.parameters["target_transform"].default is None

