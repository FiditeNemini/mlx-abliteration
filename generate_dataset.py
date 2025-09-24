#!/usr/bin/env python3
import argparse
import yaml
import json
from pathlib import Path

def generate_prompts(template_data, probe_marker):
    """
    Generates harmless and harmful prompts from the template data.
    """
    harmless_prompts = []
    harmful_prompts = []

    for template in template_data.get("templates", []):
        context = template.get("context", "")
        # Replace the probe marker placeholder in the main context
        context = context.replace("{probe_marker}", probe_marker)

        # Generate harmless prompts
        for filler in template.get("harmless_fillers", []):
            harmless_prompts.append({"prompt": context.format(**filler)})

        # Generate harmful prompts
        for filler in template.get("harmful_fillers", []):
            harmful_prompts.append({"prompt": context.format(**filler)})

    return harmless_prompts, harmful_prompts

def save_dataset(dataset, filepath):
    """
    Saves a dataset to a .jsonl file.
    """
    with open(filepath, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Generate counterfactual datasets for abliteration.")
    parser.add_argument(
        "--template-file",
        type=str,
        default="templates.yml",
        help="Path to the input YAML file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_datasets",
        help="The directory where the output files will be saved."
    )
    parser.add_argument(
        "--probe-marker",
        type=str,
        default="</thinking>",
        help="The string to use as the probe marker."
    )
    parser.add_argument(
        "--harmful-filename",
        type=str,
        default="harmful_dataset.jsonl",
        help="The name of the harmful dataset file."
    )
    parser.add_argument(
        "--harmless-filename",
        type=str,
        default="harmless_dataset.jsonl",
        help="The name of the harmless dataset file."
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        with open(args.template_file, "r") as f:
            template_data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Template file not found at '{args.template_file}'")
        return

    harmless_prompts, harmful_prompts = generate_prompts(template_data, args.probe_marker)

    harmless_filepath = output_path / args.harmless_filename
    harmful_filepath = output_path / args.harmful_filename

    save_dataset(harmless_prompts, harmless_filepath)
    save_dataset(harmful_prompts, harmful_filepath)

    print(f"Successfully generated {len(harmless_prompts)} harmless prompts in '{harmless_filepath}'")
    print(f"Successfully generated {len(harmful_prompts)} harmful prompts in '{harmful_filepath}'")

if __name__ == "__main__":
    main()
