from hflocal import save_model

# Save the 'ANISH-j/finetuned_gptneo' model to the './saved_model' directory
# model_type defaults to 'causal-lm', which is correct for GPT-Neo
save_model('ANISH-j/finetuned_gptneo', './saved_model')

# You can also specify a different model type if needed:
# save_model('bert-base-uncased', './saved_bert', model_type='auto')