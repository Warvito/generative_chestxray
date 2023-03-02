# Latent Diffusion Models for Chest X-Ray Generation using MONAI Generative Models

We use https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv-cxr/txt/create_section_files.py to create the text
sections files.

List of execution for preprocessing:
preprocessing/organise.py - resize dataset to 512
preprocessing/create_ids.py - create path files for training
preprocessing/create_section_files.py - create section files
python -m spacy download en_core_web_sm
preprocessing/create_sentences_files.py - create sentence files



TODO LIST:

- [ ] Add original implementation from adversarial training.
- [ ] Test with Microsoft and Stable Diffusion text part.
- [ ] Add synthetic sentences based on other source of information
