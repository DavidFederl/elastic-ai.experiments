import torch

from src.nn.delta.consecutive import compress, inflate


def test_compress_basic_functionality():
    """Test basic compression functionality."""
    # Test with simple data
    data = torch.tensor([10, 20, 30, 40, 50], dtype=torch.float32)
    bit_width = 8

    compressed = compress(data, bit_width)

    # First value should remain the same
    assert compressed[0] == data[0]

    # Subsequent values should be deltas
    expected_deltas = torch.tensor([10, 10, 10, 10, 10], dtype=torch.float32)
    assert torch.allclose(compressed, expected_deltas)


def test_inflate_basic_functionality():
    """Test basic inflation functionality."""
    # Test with simple delta data
    delta = torch.tensor([10, 10, 10, 10, 10], dtype=torch.float32)
    bit_width = 8

    inflated = inflate(delta, bit_width)

    # Should reconstruct original data
    expected_data = torch.tensor([10, 20, 30, 40, 50], dtype=torch.float32)
    assert torch.allclose(inflated, expected_data)


def test_compress_inflate_roundtrip():
    """Test that compress followed by inflate returns original data."""
    # Test with various data patterns
    test_cases = [
        torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32),  # Linear
        torch.tensor([5, 5, 5, 5, 5], dtype=torch.float32),  # Constant
        torch.tensor([1, 3, 6, 10, 15], dtype=torch.float32),  # Quadratic
        torch.randn(100),  # Random
    ]

    bit_width = 8

    for original_data in test_cases:
        compressed = compress(original_data, bit_width)
        inflated = inflate(compressed, bit_width)

        # Should be close to original (within clamping limits)
        assert torch.allclose(inflated, original_data, atol=1e-5)


def test_compress_clamping():
    """Test that compression clamps values to bit width limits."""
    # Create data with large deltas that should be clamped
    data = torch.tensor([0, 1000, -500, 2000, -1000], dtype=torch.float32)
    bit_width = 8

    compressed = compress(data, bit_width)

    # First value should remain unchanged
    assert compressed[0] == data[0]

    # Check that deltas are clamped to 8-bit signed range [-128, 127]
    min_val = -(2 ** (bit_width - 1))  # -128
    max_val = (2 ** (bit_width - 1)) - 1  # 127

    # All deltas except first should be within range
    assert torch.all((compressed[1:] >= min_val) & (compressed[1:] <= max_val))


def test_inflate_clamping():
    """Test that inflation clamps values to bit width limits."""
    # Create delta data with values that should be clamped during inflation
    delta = torch.tensor([0, 200, -150, 300, -200], dtype=torch.float32)
    bit_width = 8

    inflated = inflate(delta, bit_width)

    # Check that reconstructed values are clamped to 8-bit signed range
    min_val = -(2 ** (bit_width - 1))  # -128
    max_val = (2 ** (bit_width - 1)) - 1  # 127

    # All values except first should be within range
    assert torch.all((inflated[1:] >= min_val) & (inflated[1:] <= max_val))


def test_different_bit_widths():
    """Test compression with different bit widths."""
    data = torch.tensor([0, 10, 20, 30, 40], dtype=torch.float32)

    bit_widths = [4, 8, 16, 32]

    for bit_width in bit_widths:
        compressed = compress(data, bit_width)
        inflated = inflate(compressed, bit_width)

        # Should reconstruct original data within bit width limits
        min_val = -(2 ** (bit_width - 1))
        max_val = (2 ** (bit_width - 1)) - 1

        assert torch.all((inflated[1:] >= min_val) & (inflated[1:] <= max_val))


def test_multidimensional_tensors():
    """Test that compression works with multidimensional tensors."""
    # Test with 2D tensor
    data_2d = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    bit_width = 8

    compressed = compress(data_2d, bit_width)
    inflated = inflate(compressed, bit_width)

    assert torch.allclose(inflated, data_2d)

    # Test with 3D tensor
    data_3d = torch.randn(2, 3, 4)
    compressed = compress(data_3d, bit_width)
    inflated = inflate(compressed, bit_width)

    assert torch.allclose(inflated, data_3d, atol=1e-5)


def test_edge_cases():
    """Test edge cases and special values."""
    bit_width = 8

    # Test with single element
    single_data = torch.tensor([42.0])
    compressed = compress(single_data, bit_width)
    inflated = inflate(compressed, bit_width)
    assert torch.allclose(inflated, single_data)

    # Test with zeros
    zero_data = torch.zeros(10)
    compressed = compress(zero_data, bit_width)
    inflated = inflate(compressed, bit_width)
    assert torch.allclose(inflated, zero_data)

    # Test with negative values
    neg_data = torch.tensor([-10, -20, -30, -40], dtype=torch.float32)
    compressed = compress(neg_data, bit_width)
    inflated = inflate(compressed, bit_width)
    assert torch.allclose(inflated, neg_data)


def test_data_preservation():
    """Test that data is preserved through compression/decompression cycle."""
    # Create test data with various patterns
    original_data = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=torch.float32
    )

    bit_width = 16  # Use larger bit width to minimize clamping

    compressed = compress(original_data, bit_width)
    decompressed = inflate(compressed, bit_width)

    # Should be very close to original
    assert torch.allclose(decompressed, original_data, atol=1e-6)


def test_deterministic_behavior():
    """Test that compression is deterministic."""
    data = torch.randn(100)
    bit_width = 8

    # Run compression multiple times
    result1 = compress(data, bit_width)
    result2 = compress(data, bit_width)

    # Results should be identical
    assert torch.allclose(result1, result2)

    # Same for inflation
    delta = torch.randn(100)
    inflate1 = inflate(delta, bit_width)
    inflate2 = inflate(delta, bit_width)

    assert torch.allclose(inflate1, inflate2)

