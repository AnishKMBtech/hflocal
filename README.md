Happy to learn writing a pypi library and pushing it


# hflocal

`hflocal` is a Python library designed to simplify the process of saving, loading, and using Hugging Face models locally. This library provides a user-friendly interface for handling pre-trained models from the Hugging Face repository.

## Author

**Anish KM**
- GitHub: [@AnishKMBtech](https://github.com/AnishKMBtech)

## Features

- Save pre-trained models and tokenizers to a local directory
- Load pre-trained models and tokenizers from a local directory
- Use models with a simple pipeline interface for various NLP tasks

## Installation

You can install the `hflocal` library using pip:

```bash
pip install hflocal
```

## Usage

### Saving a Model

To save a pre-trained model and its tokenizer to a local directory, use the `save_model` function:

```python
from hflocal import save_model

# Save the 'bert-base-uncased' model to the './saved_model' directory
save_model('bert-base-uncased', './saved_model')
```

### Loading a Model

To load a pre-trained model and its tokenizer from a local directory, use the `load_model` function:

```python
from hflocal import load_model

# Load the model and tokenizer from the './saved_model' directory
model, tokenizer = load_model('./saved_model')
```

### Using the Model Pipeline

To use a model with a simple pipeline interface for various NLP tasks, use the `ModelPipeline` class:

```python
from hflocal import ModelPipeline

# Initialize the pipeline with the saved model
pipeline = ModelPipeline('./saved_model')

# Use the pipeline for inference
result = pipeline("Your input text here")
print(result)
```

## Development

### Setting Up the Development Environment

1. Clone the repository:
```bash
git clone https://github.com/AnishKMBtech/hflocal.git
cd hflocal
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# On Unix/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- Hugging Face for providing the pre-trained models and tokenizers
- Transformers library for the model and tokenizer implementations