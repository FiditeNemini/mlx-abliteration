# Refusal Direction Calculation Methods

The MLX Abliteration Toolkit now supports two methods for calculating the refusal direction vector:

## 1. Difference Method (Default)

The **difference** method uses the simple difference between harmful and harmless mean activations:

```
refusal_dir = harmful_mean - harmless_mean
```

This is the original method and remains the default for backward compatibility.

### CLI Usage
```bash
python cli.py -m model_path -o output_dir --refusal-dir-method difference
# or omit the flag to use the default
python cli.py -m model_path -o output_dir
```

### GUI Usage
Select "difference" from the "Refusal Direction Method" dropdown (default selection).

## 2. Projected Method

The **projected** method removes the harmless component from the refusal direction:

```
harmful_mean ─┐
              └→ refusal_dir = harmful_mean - harmless_mean
harmless_mean ─┘
       ↓
   Normalize → harmless_normalized = harmless_mean / ||harmless_mean||
       ↓
   Project → projection_scalar = refusal_dir · harmless_normalized
       ↓
   Subtract projection → refined_refusal_dir = refusal_dir - projection_scalar × harmless_normalized
```

This method ensures the refusal direction is orthogonal to the harmless activation direction, potentially providing a "cleaner" refusal vector that focuses purely on the harmful component.

### CLI Usage
```bash
python cli.py -m model_path -o output_dir --refusal-dir-method projected
```

### GUI Usage
Select "projected" from the "Refusal Direction Method" dropdown in the Advanced Options section.

## When to Use Each Method

### Use Difference (default) when:
- You want backward compatibility with previous results
- The harmless and harmful activations are well-separated
- You're following established abliteration research methodology

### Use Projected when:
- You want to isolate the purely harmful component
- The harmless activations have a strong directional component that might interfere
- You're experimenting with refined ablation techniques

## Technical Details

The projected method mathematically removes any component of the refusal direction that aligns with the harmless mean. This produces a vector that is orthogonal (perpendicular) to the harmless direction, ensuring that only the distinctly harmful features remain in the ablation target.

The projection operation is:
```
projection = (refusal_dir · harmless_normalized) × harmless_normalized
refined_refusal_dir = refusal_dir - projection
```

Where:
- `·` denotes dot product
- `×` denotes scalar multiplication
- `harmless_normalized` is the unit vector in the direction of `harmless_mean`

## Logging

When using the projected method, the toolkit logs the projection scalar value, which indicates how much of the refusal direction was aligned with the harmless direction. This information appears in the JSON logs at `~/.mlx-llm/abliteration-toolkit-cli/log.jsonl`.
