# -----------------------------------------------------------------------
# hflocal - Load a previously saved model from a local directory
#
# Prerequisites:
#   pip install torch torchvision torchaudio hflocal
#
# Run save.py first to download the model, then this script to load it.
# -----------------------------------------------------------------------

from hflocal import load_model

# Load the model and tokenizer from the './saved_model' directory
model, tokenizer = load_model('./saved_model')

# Now you can use the model for inference:
# inputs = tokenizer("Hello world", return_tensors="pt")
# outputs = model(**inputs)