import argparse
import json
from pathlib import Path
import filecmp
import mlx.core as mx
from safetensors import safe_open

# ANSI escape codes for colored output
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{bcolors.HEADER}{bcolors.BOLD}--- {text} ---{bcolors.ENDC}")

def print_ok(text):
    print(f"  ✅ {bcolors.OKGREEN}{text}{bcolors.ENDC}")

def print_fail(text):
    print(f"  ❌ {bcolors.FAIL}{text}{bcolors.ENDC}")

def print_warn(text):
    print(f"  ⚠️ {bcolors.WARNING}{text}{bcolors.ENDC}")

def print_info(text):
    print(f"  ℹ️ {bcolors.OKCYAN}{text}{bcolors.ENDC}")

def find_safetensors_file(directory: Path) -> Path:
    """Finds the safetensors file in a directory."""
    files = list(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors file found in {directory}")
    if len(files) > 1:
        print_warn(f"Multiple .safetensors files found in {directory}. Using the first one: {files[0].name}")
    return files[0]

def compare_json_files(path1: Path, path2: Path):
    """Loads and compares two JSON files."""
    try:
        with open(path1, 'r') as f1, open(path2, 'r') as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)
        if data1 == data2:
            print_ok(f"JSON files '{path1.name}' are identical.")
        else:
            print_fail(f"JSON files '{path1.name}' have differing content.")
            # Optional: Add detailed key/value diffing here if needed
    except Exception as e:
        print_fail(f"Could not compare JSON file '{path1.name}'. Error: {e}")

def compare_directories(dir1: Path, dir2: Path):
    """Compares the file listings and contents of two directories."""
    print_header("Step 1: Comparing Directory Contents")
    files1 = {p.name for p in dir1.iterdir() if p.is_file()}
    files2 = {p.name for p in dir2.iterdir() if p.is_file()}

    if files1 == files2:
        print_ok("File listings are identical.")
    else:
        print_fail("File listings are different.")
        missing_in_dir2 = files1 - files2
        added_in_dir2 = files2 - files1
        if missing_in_dir2:
            print_info(f"Files missing in new model directory: {', '.join(missing_in_dir2)}")
        if added_in_dir2:
            print_info(f"Files added in new model directory: {', '.join(added_in_dir2)}")

    common_files = files1.intersection(files2)
    for filename in sorted(list(common_files)):
        path1 = dir1 / filename
        path2 = dir2 / filename
        if filename.endswith(".json"):
            compare_json_files(path1, path2)
        elif ".safetensors" not in filename:
            # Simple binary comparison for other files (like tokenizer.model)
            if filecmp.cmp(path1, path2, shallow=False):
                print_ok(f"Files '{filename}' are identical.")
            else:
                print_fail(f"Files '{filename}' have differing content.")

def inspect_safetensors(source_path: Path, new_path: Path):
    """Performs a deep inspection of two safetensors files."""
    print_header("Step 2: Inspecting Safetensors Metadata")

    try:
        source_tensors = {}
        with safe_open(source_path, framework="mlx") as f:
            source_metadata = f.metadata()
            for key in f.keys():
                source_tensors[key] = f.get_tensor(key) # This is just metadata here
        
        new_tensors = {}
        with safe_open(new_path, framework="mlx") as f:
            new_metadata = f.metadata()
            for key in f.keys():
                new_tensors[key] = f.get_tensor(key)

        # The most likely culprit for "format: null" is the top-level metadata
        if source_metadata and new_metadata and source_metadata == new_metadata:
            print_ok("Top-level metadata (__metadata__) is identical.")
        elif source_metadata and not new_metadata:
            print_fail("New model is MISSING top-level metadata (__metadata__) that exists in the source.")
        elif not source_metadata and new_metadata:
            print_warn("New model has top-level metadata (__metadata__) but the source does not.")
        else:
            print_fail("Top-level metadata (__metadata__) is DIFFERENT.")
            print_info(f"Source metadata: {source_metadata}")
            print_info(f"New model metadata: {new_metadata}")

        print_header("Step 3: Inspecting Tensor Keys and Shapes")
        source_keys = set(source_tensors.keys())
        new_keys = set(new_tensors.keys())

        if source_keys == new_keys:
            print_ok("Tensor key sets are identical.")
        else:
            print_fail("Tensor key sets are different.")
            missing_in_new = source_keys - new_keys
            added_in_new = new_keys - source_keys
            if missing_in_new:
                print_info(f"Tensor keys missing in new model: {', '.join(missing_in_new)}")
            if added_in_new:
                print_info(f"Tensor keys added in new model: {', '.join(added_in_new)}")

        print_header("Step 4: Comparing Individual Tensor Properties (dtype and shape)")
        common_keys = sorted(list(source_keys.intersection(new_keys)))
        mismatches_found = 0
        for key in common_keys:
            source_info = source_tensors[key]
            new_info = new_tensors[key]

            shape_match = (source_info.shape == new_info.shape)
            dtype_match = (source_info.dtype == new_info.dtype)

            if not shape_match or not dtype_match:
                mismatches_found += 1
                print_fail(f"Mismatch for tensor '{key}':")
                if not shape_match:
                    print_info(f"  - Shape: Source={source_info.shape} vs New={new_info.shape}")
                if not dtype_match:
                    print_info(f"  - Dtype: Source={source_info.dtype} vs New={new_info.dtype}")

        if mismatches_found == 0:
            print_ok("All common tensors have matching dtypes and shapes.")
        else:
            print_fail(f"Found {mismatches_found} tensors with mismatched properties.")

    except Exception as e:
        print_fail(f"An error occurred during safetensors inspection: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Diagnose differences between two MLX model directories, focusing on safetensors issues.")
    parser.add_argument("source_model_dir", type=str, help="Path to the original (source) model directory.")
    parser.add_argument("new_model_dir", type=str, help="Path to the new (abliterated) model directory.")
    args = parser.parse_args()

    source_dir = Path(args.source_model_dir)
    new_dir = Path(args.new_model_dir)

    if not source_dir.is_dir() or not new_dir.is_dir():
        print_fail("One or both provided paths are not valid directories.")
        return

    # Run Comparisons
    compare_directories(source_dir, new_dir)
    try:
        source_sf_path = find_safetensors_file(source_dir)
        new_sf_path = find_safetensors_file(new_dir)
        inspect_safetensors(source_sf_path, new_sf_path)
    except FileNotFoundError as e:
        print_fail(str(e))

if __name__ == "__main__":
    main()
