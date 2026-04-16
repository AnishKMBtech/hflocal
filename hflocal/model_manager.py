import os
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
)

# Map of friendly names to Auto classes for common model types
MODEL_CLASS_MAP = {
    "auto": AutoModel,
    "causal-lm": AutoModelForCausalLM,
    "seq2seq": AutoModelForSeq2SeqLM,
    "sequence-classification": AutoModelForSequenceClassification,
    "token-classification": AutoModelForTokenClassification,
    "question-answering": AutoModelForQuestionAnswering,
}


def save_model(model_name, save_directory, model_type="causal-lm"):
    """
    Download a pre-trained model and tokenizer from Hugging Face and save them locally.

    Args:
        model_name (str): The Hugging Face model identifier (e.g. 'gpt2', 'bert-base-uncased').
        save_directory (str): The local folder path where the model and tokenizer will be saved.
        model_type (str): The type of model to download. Options:
            'auto', 'causal-lm', 'seq2seq', 'sequence-classification',
            'token-classification', 'question-answering'.
            Defaults to 'causal-lm'.

    Raises:
        ValueError: If model_type is not recognized.
        Exception: If the model download or save fails.
    """
    # Validate model type
    if model_type not in MODEL_CLASS_MAP:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from: {', '.join(MODEL_CLASS_MAP.keys())}"
        )

    # Create the save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    model_class = MODEL_CLASS_MAP[model_type]

    try:
        print(f"Downloading model '{model_name}' (type: {model_type})...")
        model = model_class.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download model '{model_name}'. "
            f"Check that the model name is correct and you have internet access.\n"
            f"Original error: {e}"
        ) from e

    try:
        print(f"Downloading tokenizer for '{model_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download tokenizer for '{model_name}'.\n"
            f"Original error: {e}"
        ) from e

    try:
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        print(f"Model and tokenizer saved to '{save_directory}'")
    except Exception as e:
        raise RuntimeError(
            f"Failed to save model/tokenizer to '{save_directory}'.\n"
            f"Original error: {e}"
        ) from e


def load_model(save_directory, model_type="causal-lm"):
    """
    Load a pre-trained model and tokenizer from a local directory.

    Args:
        save_directory (str): The directory where the model and tokenizer are saved.
        model_type (str): The type of model to load. Must match the type used during save.
            Defaults to 'causal-lm'.

    Returns:
        tuple: (model, tokenizer) — the loaded model and tokenizer.

    Raises:
        FileNotFoundError: If the save directory does not exist.
        ValueError: If model_type is not recognized.
    """
    if not os.path.isdir(save_directory):
        raise FileNotFoundError(
            f"Directory '{save_directory}' does not exist. "
            f"Did you run save_model() first?"
        )

    if model_type not in MODEL_CLASS_MAP:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from: {', '.join(MODEL_CLASS_MAP.keys())}"
        )

    model_class = MODEL_CLASS_MAP[model_type]

    try:
        print(f"Loading model from '{save_directory}'...")
        model = model_class.from_pretrained(save_directory)
        tokenizer = AutoTokenizer.from_pretrained(save_directory)
        print("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model from '{save_directory}'.\n"
            f"Original error: {e}"
        ) from e