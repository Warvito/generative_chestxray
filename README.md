# Latent Diffusion Models for Chest X-Ray Generation using MONAI Generative Models

After downloading the JPG images from [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) and the
associated free-text reports from [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.0.0/), you need to
preprocess the data. The following is the list of execution for preprocessing:
1) preprocessing/organise.py - Resizes dataset to 512 pixels in the smaller dimension
2) preprocessing/create_ids.py - Create files with datalist for training, validation and test using only "PA" views
3) preprocessing/create_section_files.py - Create file with text sections for each report.
4) preprocessing/create_sentences_files.py - Create file with sentences for each report.


TODO LIST:

- [ ] Add original implementation from adversarial training.
- [ ] Test with Microsoft and Stable Diffusion text part.
- [ ] Add synthetic sentences based on other source of information
- [ ] Maybe use LLM to augment the reports
- [ ] Add warmup time for the diffusion model
- [ ] Improve lr scheduler
- [ ] Use EMA in the diffusion model training
- [ ] Include images from ChestX-ray14 https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345
