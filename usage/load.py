#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#your looking at the code implementation of the hflocal library to load a model from hugging face to the local directory
#install the pytorch first by using pip3 install torch torchvision torchaudio
#install by using pip install hflocal
#or else install in single command by pip install torch torchvision torchaudio hflocal

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#regular implementation of hugging face libraries 
'''
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer from the local directory
tokenizer = AutoTokenizer.from_pretrained("./localmodel")
model = AutoModelForCausalLM.from_pretrained("./localmodel")
'''

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


from hflocal import load_model

# Load the model and tokenizer from the './saved_model' directory
model, tokenizer = load_model('./saved_model')

#-----------------------------------------------------------------------------------END-----------------------------------------------------------------------------------------------