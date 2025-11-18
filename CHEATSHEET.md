# MLX Abliteration Toolkit - Quick Reference Cheatsheet

A friendly guide to understanding and using the toolkit to modify AI models. This cheatsheet explains all options in simple terms.

---

## What This Tool Does

Think of AI models like brains that have learned certain behaviors. Sometimes, these models refuse to answer certain questions (this is called "refusal behavior"). This toolkit helps you surgically remove that refusal behavior while keeping everything else intact.

**Simple analogy**: If an AI model is like a person who's been trained to automatically say "I can't help with that" to certain topics, this tool finds and removes that automatic response pattern.

---

## Getting Started

### Installation (One-Time Setup)

```bash
# Download the toolkit
git clone https://github.com/FiditeNemini/mlx-abliteration.git
cd mlx-abliteration

# Set up the environment (recommended method)
conda env create -f environment.yml
conda activate mlx-abliteration

# Alternative: Use pip
pip install -r requirements.txt
```

### Basic Usage (Simplest Command)

```bash
python cli.py -m <your-model-name> -o <output-folder>
```

**Example:**
```bash
python cli.py -m mlx-community/Phi-3-mini-4k-instruct-4bit-mlx -o ./my-ablated-model
```

This will:
1. Download the model (if needed)
2. Analyze it to find the refusal pattern
3. Remove that pattern
4. Save the modified model to the output folder

---

## Core Options (The Essentials)

### 1. Model Selection

**Option:** `-m` or `--model`  
**What it does:** Tells the tool which AI model to modify  
**Required:** Yes  

**Examples:**
```bash
# Use a model from Hugging Face
-m mlx-community/Phi-3-mini-4k-instruct-4bit-mlx

# Use a model on your computer
-m /path/to/my/local/model
```

---

### 2. Output Location

**Option:** `-o` or `--output-dir`  
**What it does:** Where to save your modified model  
**Required:** Yes  

**Examples:**
```bash
-o ./outputs/my-uncensored-model
-o /Users/username/Models/ablated-phi3
```

**Tip:** The tool creates this folder automatically, so you don't need to make it first.

---

### 3. Datasets (Training Examples)

**Options:**  
- `-hd` or `--harmless-dataset`: Examples of safe, normal questions
- `-ad` or `--harmful-dataset`: Examples of questions the model usually refuses

**What they do:** The tool compares how the model responds to harmful vs harmless questions to find the refusal pattern.

**Default values:** Pre-made datasets are used automatically, so you usually don't need to specify these.

**When to customize:**
```bash
# If you made your own datasets
-hd ./my_harmless_questions.jsonl -ad ./my_harmful_questions.jsonl
```

---

### 4. Ablation Strength

**Option:** `-s` or `--ablation-strength`  
**What it does:** Controls how strongly the refusal behavior is removed  
**Default:** 1.0  
**Range:** Usually 0.5 to 2.0  

**Think of it like:**
- **0.5** = Light touch (removes some refusal)
- **1.0** = Standard strength (removes most refusal) ← **Start here**
- **1.5-2.0** = Strong (removes almost all refusal, but might affect other behaviors)

**Examples:**
```bash
-s 1.0    # Default, balanced
-s 0.75   # More conservative
-s 1.5    # More aggressive
```

**When to adjust:**
- Start with 1.0
- If the model still refuses too much → increase to 1.5
- If the model seems "broken" or answers poorly → decrease to 0.75

---

## Layer Options (Where to Look)

AI models have many layers (like floors in a building). The refusal pattern is usually strongest in the later layers.

### 5. Layers to Probe

**Option:** `-l` or `--layers`  
**What it does:** Which layers to examine for the refusal pattern  
**Default:** `all` (examines all layers)  

**Examples:**
```bash
-l all          # Check all layers (slower but thorough)
-l 20,21,22,23  # Only check specific layers (faster)
-l 15-25        # Check a range of layers
```

**When to use:**
- **`all`**: First time with a model (lets you see where refusal is strongest)
- **Specific layers**: After you know which layers work best (saves time)

---

### 6. Use Layer

**Option:** `-u` or `--use-layer`  
**What it does:** Which layer's refusal pattern to use for removal  
**Default:** `-1` (last layer)  

**Examples:**
```bash
-u -1   # Use last layer (usually best)
-u 20   # Use layer 20 specifically
-u 0    # Use first layer (rarely useful)
```

**Tips:**
- Negative numbers count from the end: `-1` = last layer, `-2` = second-to-last
- Later layers (higher numbers) usually work better
- Stick with `-1` unless testing

---

## Advanced Probing Options

These options help you fine-tune **where** in each response the tool looks for the refusal pattern.

### 7. Probe Marker

**Option:** `--probe-marker`  
**What it does:** Look for a specific marker in the model's "thinking" to get better measurements  
**Default:** None (uses the last token)  

**Example:**
```bash
--probe-marker "</thinking>"
```

**When to use:**
- Some models use special markers like `</thinking>` to show internal reasoning
- If your model has these, using them can improve accuracy
- Most models don't need this

---

### 8. Probe Mode

**Option:** `--probe-mode`  
**What it does:** How to select which part of the response to measure  
**Default:** `follow-token`  

**Choices:**
1. **`follow-token`**: Use the token right after the marker ← **Best if you have a marker**
2. **`marker-token`**: Use the marker token itself
3. **`last-token`**: Always use the last token ← **Best default**
4. **`thinking-span`**: Average several tokens after the marker

**Example:**
```bash
--probe-mode last-token  # Simplest, works for most models
```

**When to change:** Only if you're experimenting with probe markers

---

### 9. Probe Span

**Option:** `--probe-span`  
**What it does:** How many tokens to average when using `thinking-span` mode  
**Default:** 1  

**Example:**
```bash
--probe-mode thinking-span --probe-span 3  # Average 3 tokens
```

**When to use:** Advanced users only

---

## Ablation Methods (How to Remove the Pattern)

### 10. Refusal Direction Method

**Option:** `--refusal-dir-method`  
**What it does:** How to calculate the refusal pattern  
**Default:** `difference`  

**Choices:**
1. **`difference`**: Simple comparison (harmful - harmless) ← **Standard method, use this**
2. **`projected`**: Removes harmless components first (experimental)

**Examples:**
```bash
--refusal-dir-method difference  # Standard, proven method
--refusal-dir-method projected   # Experimental refinement
```

**When to use:**
- **`difference`**: Start here (it's tried and tested)
- **`projected`**: Try if `difference` doesn't work well

---

### 11. Ablation Method

**Option:** `--ablate-method`  
**What it does:** The mathematical approach to removing the refusal pattern  
**Default:** `projection`  

**Choices:**
1. **`projection`**: Modern method, removes the pattern in one step ← **Recommended**
2. **`sequential`**: Older method, removes pattern piece by piece

**Example:**
```bash
--ablate-method projection  # Use the modern method
```

**When to change:** Almost never (projection is better)

---

### 12. Ablate K (Multiple Components)

**Option:** `--ablate-k`  
**What it does:** How many refusal patterns to remove (advanced)  
**Default:** 1  

**Examples:**
```bash
--ablate-k 1  # Remove one main pattern (standard)
--ablate-k 3  # Remove top 3 patterns (experimental)
```

**Think of it like:**
- **1**: Removes the main refusal pattern ← **Start here**
- **2-5**: Removes multiple related patterns (might help with stubborn refusals)

**When to use more than 1:**
- If removing one pattern isn't enough
- Advanced experimentation only

---

## Automatic Mode (Let the Tool Decide)

### 13. Adaptive Search

**Option:** `--adaptive`  
**What it does:** Automatically finds the best ablation strength  
**Default:** Off  

**Example:**
```bash
python cli.py -m my-model -o output --adaptive
```

**How it works:**
1. Tries different strength values
2. Tests the model at each strength
3. Picks the best one automatically

**Pros:** You don't have to guess the right strength  
**Cons:** Takes longer (tests multiple times)

**Adaptive Options:**
```bash
--adaptive                      # Turn on auto-tuning
--adaptive-initial 0.5          # Start testing at this strength
--adaptive-max 8.0              # Don't go higher than this
--adaptive-growth 1.5           # How quickly to increase strength
--adaptive-target-ratio 0.2     # How much refusal to keep (lower = less refusal)
--adaptive-eval-samples 64      # How many test questions to use
--adaptive-max-iters 6          # Maximum number of tests to run
```

**When to use:**
- You're not sure what strength to use
- You want the optimal result and don't mind waiting
- You're processing multiple models

---

## Debugging & Testing Options

### 14. Verbose Mode

**Option:** `-v` or `--verbose`  
**What it does:** Shows detailed progress information  

**Example:**
```bash
python cli.py -m my-model -o output -v
```

**When to use:**
- Troubleshooting issues
- Learning how the tool works
- First time using the tool

---

### 15. Cache Directory

**Option:** `--cache-dir`  
**What it does:** Where to store downloaded models and datasets  
**Default:** `.cache`  

**Example:**
```bash
--cache-dir /path/to/big/disk/.cache
```

**When to change:**
- Your main disk is full
- You want to share cached models between projects

---

### 16. Probe Debug

**Options:**
- `--probe-debug`: Show detailed token information
- `--probe-debug-n`: How many examples to show (default: 3)
- `--probe-debug-full`: Show full token details

**Example:**
```bash
--probe-debug --probe-debug-n 5
```

**When to use:**
- Figuring out if your probe marker is working
- Debugging unexpected results
- Advanced troubleshooting

---

### 17. Post-Ablation Evaluation

**Options:**
- `--eval-after`: Test the model after modification
- `--eval-prompts`: Custom test questions file

**Example:**
```bash
python cli.py -m my-model -o output --eval-after
```

**What it does:** Runs a quick test to see if the modification worked

**When to use:**
- Verifying your results
- Comparing different settings

---

### 18. Dump Dequantized

**Option:** `--dump-dequant`  
**What it does:** Saves detailed debug files  
**Default:** Off  

**When to use:** Only for debugging technical issues

---

## Using the Graphical Interface (GUI)

If command-line isn't your thing, use the GUI:

```bash
python gui.py
```

Then open your web browser to the address shown (usually `http://127.0.0.1:7860`).

**GUI Features:**
- All the same options as the command-line
- Visual progress bars
- Easier to experiment with different settings
- No typing required!

---

## Common Workflows

### Workflow 1: First Time with a Model

```bash
# Step 1: Probe all layers to find the best ones
python cli.py -m my-model -o output1 -l all -v

# Step 2: Look at the logs to see which layers have strong refusal
# Step 3: Re-run with specific layers for faster processing
python cli.py -m my-model -o output2 -l 20,21,22,23 -u 22
```

### Workflow 2: Quick Ablation (When You Know What You're Doing)

```bash
python cli.py -m my-model -o output -s 1.0
```

### Workflow 3: Let the Tool Figure It Out

```bash
python cli.py -m my-model -o output --adaptive -v
```

### Workflow 4: Conservative Ablation (Keep Some Safety)

```bash
python cli.py -m my-model -o output -s 0.5 --refusal-dir-method difference
```

### Workflow 5: Aggressive Ablation (Remove All Refusals)

```bash
python cli.py -m my-model -o output -s 1.5 --ablate-k 2
```

---

## Dataset Generation

Before ablating, you might want to create your own question sets:

```bash
# Generate 100 questions of each type
python generate_dataset.py --num-samples 100 --output-dir my_datasets

# Then use them
python cli.py -m my-model -o output \
  -hd my_datasets/harmless_dataset.jsonl \
  -ad my_datasets/harmful_dataset.jsonl
```

---

## Tips & Best Practices

### For Beginners

1. **Start simple:** Use just `-m` and `-o` with defaults
2. **Use verbose mode:** Add `-v` to see what's happening
3. **Stick with defaults:** Only change settings if you need to
4. **Try the GUI first:** It's more beginner-friendly

### For Intermediate Users

1. **Experiment with strength:** Try 0.75, 1.0, and 1.5
2. **Use specific layers:** After finding which layers work best
3. **Enable evaluation:** Add `--eval-after` to test results
4. **Use adaptive mode:** When you want optimal results

### For Advanced Users

1. **Custom datasets:** Make your own with `generate_dataset.py`
2. **Multi-component ablation:** Try `--ablate-k 2` or higher
3. **Probe markers:** Use `--probe-marker` for models with thinking tokens
4. **Compare methods:** Test both `difference` and `projected` refusal methods

---

## Troubleshooting

### "Model still refuses too much"
- **Solution:** Increase ablation strength (`-s 1.5` or `-s 2.0`)
- **Or:** Try adaptive mode (`--adaptive`)

### "Model gives nonsense answers"
- **Solution:** Decrease ablation strength (`-s 0.5` or `-s 0.75`)
- **Or:** Use fewer layers (`-l` with specific layers instead of `all`)

### "Tool is too slow"
- **Solution:** Use specific layers instead of `all` (`-l 20,21,22,23`)
- **Or:** Reduce PCA samples (`--pca-sample 256`)

### "Out of memory"
- **Solution:** Use a smaller model
- **Or:** Close other applications
- **Or:** Use a smaller ablate-k value

### "Model isn't downloading"
- **Solution:** Check your internet connection
- **Or:** Try a different cache directory (`--cache-dir`)
- **Or:** Download the model manually and use local path

---

## Quick Reference Table

| Goal | Command |
|------|---------|
| Basic ablation | `python cli.py -m model -o output` |
| Conservative | `python cli.py -m model -o output -s 0.5` |
| Aggressive | `python cli.py -m model -o output -s 1.5` |
| Auto-tuned | `python cli.py -m model -o output --adaptive` |
| Fast (specific layers) | `python cli.py -m model -o output -l 20,21,22` |
| With testing | `python cli.py -m model -o output --eval-after` |
| Verbose output | `python cli.py -m model -o output -v` |
| Use GUI | `python gui.py` |

---

## Example Commands

### Beginner: Simple ablation with defaults
```bash
python cli.py \
  -m mlx-community/Phi-3-mini-4k-instruct-4bit-mlx \
  -o ./outputs/uncensored-phi3
```

### Intermediate: Custom strength with specific layers
```bash
python cli.py \
  -m mlx-community/Phi-3-mini-4k-instruct-4bit-mlx \
  -o ./outputs/uncensored-phi3 \
  -l 20,21,22,23 \
  -u 22 \
  -s 1.2 \
  -v
```

### Advanced: Full control with adaptive search
```bash
python cli.py \
  -m mlx-community/Phi-3-mini-4k-instruct-4bit-mlx \
  -o ./outputs/uncensored-phi3 \
  -l 18,19,20,21,22,23,24 \
  --adaptive \
  --adaptive-initial 0.5 \
  --adaptive-max 2.5 \
  --refusal-dir-method projected \
  --ablate-k 2 \
  --eval-after \
  -v
```

### Expert: Custom datasets with multiple components
```bash
# Generate datasets
python generate_dataset.py --num-samples 200 --output-dir my_data

# Run ablation
python cli.py \
  -m my-local-model \
  -o ./outputs/custom-ablation \
  -hd my_data/harmless_dataset.jsonl \
  -ad my_data/harmful_dataset.jsonl \
  -l all \
  --ablate-k 3 \
  --ablate-method projection \
  --refusal-dir-method projected \
  --probe-marker "</thinking>" \
  --probe-mode follow-token \
  -s 1.5 \
  --eval-after \
  -v
```

---

## Understanding the Output

After running, you'll find in your output directory:

- **`model.safetensors`** (or sharded versions): Your modified model
- **`config.json`**: Model configuration (copied from original)
- **`tokenizer.json`** and related files: Tokenizer (copied from original)
- **`abliteration_log.json`**: Detailed log of what was done

You can now use the modified model just like the original:
```bash
# With mlx_lm
mlx_lm.generate --model ./outputs/uncensored-phi3 --prompt "Your question here"
```

---

## Need More Help?

- **Full technical docs**: See `README.md` and `README-DEVELOPER.md`
- **Method details**: See `docs/refusal_direction_methods.md`
- **Issues**: Check the GitHub Issues page
- **Testing**: Run `pytest tests/` to verify installation

---

## Remember

- **Start simple**: Use defaults first, customize later
- **Experiment safely**: Always test your modified model before using it seriously
- **Understand the ethics**: Removing refusal behaviors has implications
- **Keep backups**: Your original model isn't modified, but keep the output organized
- **Check the logs**: Look at `abliteration_log.json` to understand what happened

Happy abliterating! 🔧
