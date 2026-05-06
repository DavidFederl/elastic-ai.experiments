from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torchvision import transforms

from src.nn.data.smartable import (
    SmarTable,
    smartable_trainingset_flattened,
    smartable_validationset_flattened,
)


@pytest.fixture
def mock_dataset_dir(tmp_path):
    """Create a mock dataset directory structure for testing."""
    # Create SmarTable directory first
    smartable_dir = tmp_path / "SmarTable"
    smartable_dir.mkdir()

    # Create mock subjects inside SmarTable directory
    for subject_id in range(1, 4):  # 3 subjects total
        subject_dir = smartable_dir / f"subject_{subject_id}"
        subject_dir.mkdir()

        # Create mock sessions
        for session_id in range(1, 3):  # 2 sessions per subject
            session_dir = subject_dir / f"session_{session_id}"
            session_dir.mkdir()

            # Create mock classes
            for class_name in [
                "knock",
                "swipe-down",
                "swipe-left",
                "swipe-right",
                "swipe-up",
                "tap",
            ]:
                class_dir = session_dir / class_name
                class_dir.mkdir()

                # Create mock sample files in .npz format
                for sample_id in range(1, 3):  # 2 samples per class
                    sample_file = class_dir / f"sample_{sample_id}.npz"
                    # Create mock numpy data and save as .npz
                    mock_data = torch.randn(10, 10).numpy()
                    np.savez(sample_file, x=mock_data)

    return tmp_path


def test_smartable_initialization(mock_dataset_dir):
    """Test that SmarTable initializes correctly."""
    dataset = SmarTable(root=mock_dataset_dir, train=True)

    assert dataset.train
    assert dataset.classes == [
        "knock",
        "swipe-down",
        "swipe-left",
        "swipe-right",
        "swipe-up",
        "tap",
    ]
    assert len(dataset) > 0
    assert len(dataset.data) == len(dataset.targets)


def test_smartable_train_test_split(mock_dataset_dir):
    """Test that train/test split works correctly."""
    train_dataset = SmarTable(root=mock_dataset_dir, train=True)
    test_dataset = SmarTable(root=mock_dataset_dir, train=False)

    # Training set should have more samples than test set
    assert len(train_dataset) > len(test_dataset)

    # Combined, they should cover all available data
    total_samples = len(train_dataset) + len(test_dataset)
    assert 72 == total_samples

    # With 3 subjects, 80% train split should give 2 subjects for train, 1 for test
    # Each subject has 2 sessions * 6 classes * 2 samples = 24 samples
    # So train should have ~48 samples, test ~24 samples
    assert len(train_dataset) == pytest.approx(48, abs=5)
    assert len(test_dataset) == pytest.approx(24, abs=5)


def test_smartable_getitem(mock_dataset_dir):
    """Test that __getitem__ returns correct data format."""
    dataset = SmarTable(root=mock_dataset_dir, train=True)

    # Test getting a single item
    sample, target = dataset[0]

    assert isinstance(sample, torch.Tensor)
    assert isinstance(target, int)
    assert 0 <= target < len(dataset.classes)


def test_smartable_transforms(mock_dataset_dir):
    """Test that transforms are applied correctly."""
    # Mock transform that adds 1 to all values
    mock_transform = MagicMock(return_value=torch.tensor([2.0, 3.0]))

    dataset = SmarTable(
        root=mock_dataset_dir,
        train=True,
        transform=mock_transform,
        target_transform=lambda x: x * 2,
    )

    _, target = dataset[0]

    # Verify transform was called
    mock_transform.assert_called_once()

    # Verify target transform was applied
    assert target == dataset.targets[0] * 2


def test_smartable_flattened_functions():
    """Test the convenience functions for flattened datasets."""
    with patch("src.nn.data.smartable.SmarTable") as mock_smartable:
        mock_instance = MagicMock()
        mock_smartable.return_value = mock_instance

        # Test training set function
        _ = smartable_trainingset_flattened()
        # Check that SmarTable was called with correct parameters
        assert mock_smartable.call_count == 1
        call_args = mock_smartable.call_args
        assert call_args[1]["root"] == "datasets"
        assert call_args[1]["train"]
        assert isinstance(call_args[1]["transform"], transforms.Compose)

        # Test validation set function
        _ = smartable_validationset_flattened()
        assert mock_smartable.call_count == 2
        call_args = mock_smartable.call_args
        assert call_args[1]["root"] == "datasets"
        assert not call_args[1]["train"]
        assert isinstance(call_args[1]["transform"], transforms.Compose)


def test_smartable_error_handling():
    """Test error handling for invalid dataset paths."""
    with pytest.raises(RuntimeError, match="Dataset folder not found!"):
        SmarTable(root="/nonexistent/path", train=True)
