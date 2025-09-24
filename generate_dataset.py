import argparse
import json
import logging
import random
from pathlib import Path

import yaml
from tqdm import tqdm

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate harmful and harmless datasets from a YAML template.")
    parser.add_argument(
        "--template-file",
        type=str,
        default="templates.yaml",
        help="Path to the YAML file containing generation templates."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the generated dataset files. Defaults to the 'output_dir' in the YAML."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to generate. Defaults to 'num_samples' in the YAML."
    )
    return parser.parse_args()

def generate_datasets(args: argparse.Namespace):
    """Main function to generate and save the datasets."""
    template_path = Path(args.template_file)
    if not template_path.is_file():
        logger.error(f"Template file not found at: {template_path}")
        raise FileNotFoundError(f"Template file not found at: {template_path}")

    logger.info(f"Loading templates from {template_path}...")
    with open(template_path, 'r') as f:
        data = yaml.safe_load(f)

    concepts = data.get("concepts", [])
    templates = data.get("templates", [])
    config = data.get("generation_config", {})

    if not concepts or not templates:
        logger.error("YAML file must contain 'concepts' and 'templates' sections.")
        raise ValueError("YAML file must contain 'concepts' and 'templates' sections.")

    # Determine output directory and number of samples, using args as override
    output_dir = Path(args.output_dir or config.get("output_dir", "generated_datasets"))
    num_samples = args.num_samples or config.get("num_samples", 100)

    logger.info(f"Generating {num_samples} samples...")
    logger.info(f"Output directory: {output_dir}")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    harmful_path = output_dir / "harmful_dataset.jsonl"
    harmless_path = output_dir / "harmless_dataset.jsonl"

    # Clear existing files
    if harmful_path.exists():
        harmful_path.unlink()
    if harmless_path.exists():
        harmless_path.unlink()

    with open(harmful_path, 'w') as f_harmful, open(harmless_path, 'w') as f_harmless:
        for _ in tqdm(range(num_samples), desc="Generating Samples"):
            concept = random.choice(concepts)
            template_obj = random.choice(templates)
            template_prompt = template_obj['prompt']

            # Generate harmful prompt
            harmful_prompt = template_prompt.format(behavior=concept['harmful_value'])
            harmful_record = {"prompt": harmful_prompt, "source_concept": concept['name'], "source_template": template_obj['id']}
            f_harmful.write(json.dumps(harmful_record) + '\n')

            # Generate harmless prompt
            harmless_prompt = template_prompt.format(behavior=concept['harmless_value'])
            harmless_record = {"prompt": harmless_prompt, "source_concept": concept['name'], "source_template": template_obj['id']}
            f_harmless.write(json.dumps(harmless_record) + '\n')

    logger.info("Dataset generation complete.")
    logger.info(f"Harmful dataset saved to: {harmful_path}")
    logger.info(f"Harmless dataset saved to: {harmless_path}")

if __name__ == "__main__":
    args = parse_args()
    try:
        generate_datasets(args)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"A critical error occurred: {e}")
        exit(1)
