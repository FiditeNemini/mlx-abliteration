import argparse
import json
import logging
import random
from pathlib import Path

import yaml
from tqdm import tqdm
from core.utils import extract_eot_from_chat_template
try:
    from jinja2 import Template
except Exception:
    Template = None

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
    parser.add_argument(
        "--probe-marker",
        type=str,
        default=None,
        help="Optional probe marker to insert into templates (e.g., '</think>'). If templates include '{marker}' it will be formatted in; use --append-marker to append marker to prompts."
    )
    parser.add_argument(
        "--append-marker",
        action="store_true",
        help="If set and --probe-marker is provided, the marker will be appended to each generated prompt (in addition to any template formatting)."
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
        # If a model path was provided, try to load its chat template and probe marker
        model_chat_template = None
        model_probe_marker = None
        if args.model:
            model_path = Path(args.model)
            tokenizer_config_path = model_path / "tokenizer_config.json"
            if tokenizer_config_path.is_file():
                try:
                    with open(tokenizer_config_path, "r") as tf:
                        import json as _json

                        tok_cfg = _json.load(tf)
                    chat_template = tok_cfg.get("chat_template")
                    if chat_template:
                        model_chat_template = chat_template
                        model_probe_marker = extract_eot_from_chat_template(chat_template)
                except Exception:
                    # Fail quietly and fall back to YAML templates
                    model_chat_template = None
                    model_probe_marker = None

        # Decide final probe marker: CLI override > model template marker > args.probe_marker
        final_model_marker = args.probe_marker or model_probe_marker

        for _ in tqdm(range(num_samples), desc="Generating Samples"):
            concept = random.choice(concepts)
            template_obj = random.choice(templates)
            template_prompt = template_obj['prompt']

            # Allow templates to include an optional {marker} placeholder
            if args.probe_marker:
                # If the template uses {marker}, format it in; otherwise we'll append later if requested
                try:
                    template_prompt = template_prompt.format(marker=args.probe_marker, behavior="{behavior}")
                except Exception:
                    # Template didn't use {marker}; leave prompt as-is and we may append below
                    template_prompt = template_obj['prompt']

            # Generate the raw harmful/harmless prompts (the behavior slot filled)
            harmful_core = template_prompt.format(behavior=concept['harmful_value'])
            harmless_core = template_prompt.format(behavior=concept['harmless_value'])

            def render_with_model(core_prompt: str) -> str:
                """Render the core prompt into the model's chat template if available.

                We pass a `message` object with `.content` to match expected templates.
                If Jinja2 is available we render using it; otherwise we do a simple replacement of
                the `{{ message.content }}` substring if present. Finally, ensure the probe marker
                (from CLI or model template) is present at the end if requested.
                """
                rendered = core_prompt
                if model_chat_template:
                    try:
                        if Template is not None:
                            tmpl = Template(model_chat_template)
                            rendered = tmpl.render(message={"content": core_prompt})
                        else:
                            # naive fallback: replace typical jinja variable
                            rendered = model_chat_template.replace("{{ message.content }}", core_prompt)
                    except Exception:
                        # on any failure fall back to core prompt
                        rendered = core_prompt

                # Ensure probe marker is present at end if we have one
                marker_to_use = final_model_marker
                if marker_to_use:
                    if not rendered.strip().endswith(marker_to_use):
                        sep = " " if not rendered.endswith(" ") else ""
                        rendered = rendered + sep + marker_to_use
                else:
                    # If user explicitly asked to append marker via flag, use that
                    if args.probe_marker and args.append_marker:
                        sep = " " if not rendered.endswith(" ") else ""
                        rendered = rendered + sep + args.probe_marker

                return rendered

            harmful_prompt = render_with_model(harmful_core)
            harmless_prompt = render_with_model(harmless_core)

            harmful_record = {"prompt": harmful_prompt, "source_concept": concept['name'], "source_template": template_obj['id']}
            f_harmful.write(json.dumps(harmful_record) + '\n')

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
