# hflocal

Simple library to save, load, and use Hugging Face models locally.

## Installation

```bash
pip install hflocal
```

## Quick Start

**Save a model:**
```python
from hflocal import save_model

save_model('bert-base-uncased', './my_model')
```

**Load a model:**
```python
from hflocal import load_model

model, tokenizer = load_model('./my_model')
```

**Use with pipeline:**
```python
from hflocal import ModelPipeline

pipeline = ModelPipeline('./my_model')
result = pipeline("Your text here")
```

## Features

- 📥 Download and save models locally
- 📤 Load models from disk instantly
- 🚀 Simple pipeline interface for inference

## License

MIT
- Transformers library for the model and tokenizer implementations