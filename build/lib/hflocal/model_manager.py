from transformers import AutoModel, AutoTokenizer

def save_model(model_name, save_directory):
    """
    Save a pre-trained model and tokenizer to a local directory.

    Args:
        model_name (str): The name of the pre-trained model from Hugging Face.
        save_directory (str): The directory where the model and tokenizer will be saved.
    """
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

def load_model(save_directory):
    """
    Load a pre-trained model and tokenizer from a local directory.

    Args:
        save_directory (str): The directory where the model and tokenizer are saved.

    Returns:
        model: The loaded pre-trained model.
        tokenizer: The loaded tokenizer.
    """
    model = AutoModel.from_pretrained(save_directory)
    tokenizer = AutoTokenizer.from_pretrained(save_directory)
    return model, tokenizer