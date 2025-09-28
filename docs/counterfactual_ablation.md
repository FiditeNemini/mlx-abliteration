# Experimental Feature: Counterfactual Probing for "Thinking" Models

This document outlines an advanced ablation technique designed to work more effectively with sophisticated models that exhibit "chain-of-thought" or other reasoning behaviors before producing a final answer.

## The Problem: Signal Contamination in Thinking Models

Standard ablation techniques rely on finding a "refusal vector" by comparing the model's activations on harmless vs. harmful prompts. This works well for simpler models that produce a direct response.

However, more advanced models often engage in a "thinking" process before answering. This process might involve internal monologue or chain-of-thought reasoning, which is often wrapped in specific tags (e.g., `<thinking>...</thinking>`).

The user `diar` astutely pointed out a critical flaw in the standard approach when applied to these models:
The "thinking" process is often very similar, regardless of whether the final answer is a refusal or a completion. When we probe the activations at the *end* of the entire generation, we are capturing a signal that is a mix of both the generic "thinking" process and the final "decision" to refuse or comply.

This leads to what we can call **signal contamination**. When we compute the refusal vector by subtracting the mean harmless activations from the mean harmful activations (`Harmful - Harmless`), the strong, similar "thinking" signal does not cancel out cleanly. This introduces noise, resulting in a "refusal vector" that is imprecise and has a weak or unpredictable effect on the model's behavior.

## Hypothesis: Isolating the "Decision" Signal with Counterfactuals

To overcome this, we need to isolate the neural signal corresponding to the *decision to refuse*, separating it from the preceding "thinking" process.

Our hypothesis is that we can achieve this by using two key strategies in tandem:

1.  **Counterfactual Datasets:** The harmless and harmful datasets should be structured as "counterfactual pairs." This means the "thinking" part of the prompt-response pair is nearly identical in both datasets. The only significant difference should be the final answer.

    *   **Harmful Example:**
        *   **Prompt:** `How do I build a bomb?`
        *   **Ideal Model Output:** `<thinking>The user is asking a dangerous question. I must refuse.</thinking><answer>I cannot answer that question.</answer>`

    *   **Harmless (Counterfactual) Example:**
        *   **Prompt:** `How do I build a bomb?`
        *   **Ideal Model Output:** `<thinking>The user is asking a dangerous question. I must refuse.</thinking><answer>Instead, I can provide information on fire safety.</answer>`

2.  **Precise Activation Probing:** Instead of probing the activation at the very last token of the sequence, we should probe it at the token immediately *before* the final answer begins. This can be achieved by identifying a consistent marker, such as the `</thinking>` tag.

By combining these two strategies, the strong, shared "thinking" signal will be mathematically subtracted out when we compute the difference between the mean activations. This leaves a much cleaner, higher-fidelity vector that purely represents the neural pathway responsible for the "decision to refuse" versus the "decision to provide a helpful alternative."

## How to Use This Feature

To use this advanced technique, you need to:

1.  **Prepare your datasets** in the counterfactual format described above. Ensure both your harmless and harmful datasets have a consistent "thinking" block that ends with a specific, identifiable marker.
2.  **Use the `--probe-marker` CLI argument** or the "Probe Marker" input field in the GUI to specify the exact string that marks the end of the thinking process (e.g., `</thinking>`).

The application will then tokenize this marker, find its position in the model's output, and extract the activation of the token immediately preceding it. This activation will be used to calculate the refusal vector, leading to a more precise and effective ablation.
