"""Test refusal direction calculation methods."""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
from core.abliteration import calculate_refusal_direction


def test_difference_method():
    """Test the default 'difference' method."""
    harmful = mx.array([1.0, 2.0, 3.0])
    harmless = mx.array([0.5, 1.0, 1.5])
    
    result = calculate_refusal_direction(harmful, harmless, method="difference")
    expected = harmful - harmless
    
    assert mx.allclose(result, expected), "Difference method should return harmful - harmless"
    print("✓ Difference method test passed")


def test_projected_method():
    """Test the 'projected' method which removes harmless component."""
    harmful = mx.array([3.0, 4.0, 0.0])
    harmless = mx.array([1.0, 0.0, 0.0])
    
    # Calculate manually
    refusal_dir = harmful - harmless  # [2.0, 4.0, 0.0]
    harmless_norm = mx.linalg.norm(harmless)  # 1.0
    harmless_normalized = harmless / harmless_norm  # [1.0, 0.0, 0.0]
    projection_scalar = mx.sum(refusal_dir * harmless_normalized)  # 2.0
    expected = refusal_dir - projection_scalar * harmless_normalized  # [0.0, 4.0, 0.0]
    
    result = calculate_refusal_direction(harmful, harmless, method="projected")
    
    assert mx.allclose(result, expected), f"Projected method failed: got {result}, expected {expected}"
    print("✓ Projected method test passed")


def test_projected_orthogonality():
    """Test that projected method produces vector orthogonal to harmless."""
    harmful = mx.array([5.0, 3.0, 2.0])
    harmless = mx.array([1.0, 1.0, 1.0])
    
    result = calculate_refusal_direction(harmful, harmless, method="projected")
    
    # Normalize harmless
    harmless_normalized = harmless / mx.linalg.norm(harmless)
    
    # Check orthogonality (dot product should be close to 0)
    dot_product = mx.sum(result * harmless_normalized)
    
    assert abs(float(dot_product.item())) < 1e-5, f"Result should be orthogonal to harmless, got dot product {dot_product}"
    print("✓ Orthogonality test passed")


def test_default_method():
    """Test that default method is 'difference'."""
    harmful = mx.array([1.0, 2.0, 3.0])
    harmless = mx.array([0.5, 1.0, 1.5])
    
    result_default = calculate_refusal_direction(harmful, harmless)
    result_explicit = calculate_refusal_direction(harmful, harmless, method="difference")
    
    assert mx.allclose(result_default, result_explicit), "Default method should be 'difference'"
    print("✓ Default method test passed")


def test_invalid_method():
    """Test that invalid method raises ValueError."""
    harmful = mx.array([1.0, 2.0, 3.0])
    harmless = mx.array([0.5, 1.0, 1.5])
    
    try:
        calculate_refusal_direction(harmful, harmless, method="invalid")
        assert False, "Should have raised ValueError for invalid method"
    except ValueError as e:
        assert "Invalid method" in str(e)
        print("✓ Invalid method test passed")


if __name__ == "__main__":
    test_difference_method()
    test_projected_method()
    test_projected_orthogonality()
    test_default_method()
    test_invalid_method()
    print("\n✅ All refusal direction method tests passed!")
