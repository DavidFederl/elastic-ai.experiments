import json
import tempfile
from pathlib import Path

import torch

from src.nn.model.serialize import load_from_json, save_as_json


def test_save_as_json_basic_functionality():
    """Test basic JSON serialization functionality."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "test_model.json"

        # Create a simple state dict
        state_dict = {
            "layer1.weight": torch.randn(3, 3),
            "layer1.bias": torch.randn(3),
            "layer2.weight": torch.randn(2, 3),
            "layer2.bias": torch.randn(2),
        }

        # Save to JSON
        save_as_json(state_dict, path)

        # Verify file was created
        assert path.exists()
        assert path.is_file()

        # Verify JSON content
        with open(path, "r") as file:
            content = json.load(file)

        # Check that all tensors are represented
        assert len(content) == 4
        assert "layer1.weight" in content
        assert "layer1.bias" in content
        assert "layer2.weight" in content
        assert "layer2.bias" in content


def test_load_from_json_basic_functionality():
    """Test basic JSON deserialization functionality."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "test_model.json"

        # Create test JSON data
        test_data = {
            "layer1.weight": {
                "shape": [2, 2],
                "dtype": "torch.float32",
                "values": [[1.0, 2.0], [3.0, 4.0]],
            },
            "layer1.bias": {
                "shape": [2],
                "dtype": "torch.float32",
                "values": [0.1, 0.2],
            },
        }

        # Write test data to file
        with open(path, "w") as file:
            json.dump(test_data, file)

        # Load from JSON
        loaded_state_dict = load_from_json(path)

        # Verify loaded tensors
        assert len(loaded_state_dict) == 2
        assert "layer1.weight" in loaded_state_dict
        assert "layer1.bias" in loaded_state_dict

        # Verify tensor properties
        weight_tensor = loaded_state_dict["layer1.weight"]
        assert isinstance(weight_tensor, torch.Tensor)
        assert weight_tensor.shape == (2, 2)
        assert weight_tensor.dtype == torch.float32
        assert torch.allclose(weight_tensor, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

        bias_tensor = loaded_state_dict["layer1.bias"]
        assert isinstance(bias_tensor, torch.Tensor)
        assert bias_tensor.shape == (2,)
        assert bias_tensor.dtype == torch.float32
        assert torch.allclose(bias_tensor, torch.tensor([0.1, 0.2]))


def test_save_load_roundtrip():
    """Test that save followed by load returns equivalent tensors."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "roundtrip_test.json"

        # Create test state dict with various tensor types
        original_state_dict = {
            "conv.weight": torch.randn(3, 3, 3, 3),
            "conv.bias": torch.randn(3),
            "linear.weight": torch.randn(10, 20),
            "linear.bias": torch.randn(10),
            "bn.weight": torch.randn(5),
            "bn.bias": torch.randn(5),
        }

        # Save and load
        save_as_json(original_state_dict, path)
        loaded_state_dict = load_from_json(path)

        # Verify all tensors were loaded
        assert len(loaded_state_dict) == len(original_state_dict)

        # Verify tensor equivalence
        for name in original_state_dict:
            assert name in loaded_state_dict
            original_tensor = original_state_dict[name]
            loaded_tensor = loaded_state_dict[name]

            assert isinstance(loaded_tensor, torch.Tensor)
            assert loaded_tensor.shape == original_tensor.shape
            assert loaded_tensor.dtype == original_tensor.dtype
            assert torch.allclose(loaded_tensor, original_tensor, atol=1e-6)


def test_different_dtypes():
    """Test serialization with different tensor data types."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "dtypes_test.json"

        # Test various dtypes
        state_dict = {
            "float32": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            "float64": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
            "int32": torch.tensor([1, 2, 3], dtype=torch.int32),
            "int64": torch.tensor([1, 2, 3], dtype=torch.int64),
        }

        # Save and load
        save_as_json(state_dict, path)
        loaded_state_dict = load_from_json(path)

        # Verify dtypes are preserved
        assert torch.allclose(loaded_state_dict["float32"], state_dict["float32"])
        assert loaded_state_dict["float32"].dtype == torch.float32

        assert torch.allclose(loaded_state_dict["float64"], state_dict["float64"])
        assert loaded_state_dict["float64"].dtype == torch.float64

        assert torch.allclose(loaded_state_dict["int32"], state_dict["int32"])
        assert loaded_state_dict["int32"].dtype == torch.int32

        assert torch.allclose(loaded_state_dict["int64"], state_dict["int64"])
        assert loaded_state_dict["int64"].dtype == torch.int64


def test_empty_state_dict():
    """Test serialization with empty state dict."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "empty_test.json"

        # Save empty state dict
        save_as_json({}, path)

        # Load empty state dict
        loaded_state_dict = load_from_json(path)

        assert len(loaded_state_dict) == 0


def test_complex_tensor_shapes():
    """Test serialization with complex tensor shapes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "shapes_test.json"

        # Test various tensor shapes
        state_dict = {
            "1d": torch.randn(100),
            "2d": torch.randn(10, 20),
            "3d": torch.randn(5, 10, 15),
            "4d": torch.randn(3, 32, 32, 32),
            "5d": torch.randn(2, 3, 4, 5, 6),
        }

        # Save and load
        save_as_json(state_dict, path)
        loaded_state_dict = load_from_json(path)

        # Verify shapes are preserved
        assert loaded_state_dict["1d"].shape == (100,)
        assert loaded_state_dict["2d"].shape == (10, 20)
        assert loaded_state_dict["3d"].shape == (5, 10, 15)
        assert loaded_state_dict["4d"].shape == (3, 32, 32, 32)
        assert loaded_state_dict["5d"].shape == (2, 3, 4, 5, 6)


def test_invalid_dtype_handling():
    """Test handling of invalid dtype strings."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "invalid_dtype.json"

        # Create JSON with invalid dtype
        invalid_data = {
            "test": {
                "shape": [1, 1],
                "dtype": "torch.invalid_dtype",  # This dtype doesn't exist
                "values": [[1.0]],
            }
        }

        with open(path, "w") as file:
            json.dump(invalid_data, file)

        # Should return empty dict due to error
        loaded_state_dict = load_from_json(path)
        assert len(loaded_state_dict) == 0


def test_file_not_found_handling():
    """Test handling of non-existent files."""
    # Try to load from non-existent file
    path = Path("/nonexistent/path/model.json")

    # Should return empty dict
    loaded_state_dict = load_from_json(path)
    assert len(loaded_state_dict) == 0


def test_permission_error_handling():
    """Test handling of permission errors."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "readonly"
        path.mkdir()

        # Make directory read-only
        path.chmod(0o444)

        try:
            # Try to save to read-only directory
            state_dict = {"test": torch.randn(2, 2)}
            save_as_json(state_dict, path / "test.json")

            # Should not raise exception, just log error
            assert True  # If we get here, error handling worked
        finally:
            # Restore permissions for cleanup
            path.chmod(0o755)


def test_special_values():
    """Test serialization of special floating point values."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "special_values.json"

        # Test with special values
        state_dict = {
            "positive_inf": torch.tensor([float("inf")]),
            "negative_inf": torch.tensor([float("-inf")]),
            "nan": torch.tensor([float("nan")]),
            "zero": torch.tensor([0.0]),
            "negative_zero": torch.tensor([-0.0]),
        }

        # Save and load
        save_as_json(state_dict, path)
        loaded_state_dict = load_from_json(path)

        # Verify special values are handled
        assert torch.isinf(loaded_state_dict["positive_inf"]).item()
        assert torch.isinf(loaded_state_dict["negative_inf"]).item()
        assert torch.isnan(loaded_state_dict["nan"]).item()
        assert loaded_state_dict["zero"].item() == 0.0


def test_large_tensor_handling():
    """Test serialization of large tensors."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "large_test.json"

        # Create a reasonably large tensor
        large_tensor = torch.randn(100, 100)  # 10,000 elements
        state_dict = {"large": large_tensor}

        # Save and load
        save_as_json(state_dict, path)
        loaded_state_dict = load_from_json(path)

        # Verify large tensor was handled correctly
        assert torch.allclose(loaded_state_dict["large"], large_tensor, atol=1e-6)
        assert loaded_state_dict["large"].shape == (100, 100)

