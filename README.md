# Latent Diffusion Models for Chest X-Ray Generation using MONAI Generative Models

Script to train a Latent Diffusion Model based on [Pinaya et al. "Brain imaging generation with latent diffusion models.
"](https://arxiv.org/abs/2209.07162) on the MIMIC-CXR dataset using [MONAI Generative Models
](https://github.com/Project-MONAI/GenerativeModels) package.


## Instructions

After downloading the JPG images from [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) and the
associated free-text reports from [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.0.0/), you need to
preprocess the data. The following is the list of execution for preprocessing:
1) preprocessing/organise.py - Resizes dataset to 512 pixels in the smaller dimension
2) preprocessing/create_ids.py - Create files with datalist for training, validation and test using only "PA" views
3) preprocessing/create_section_files.py - Create file with text sections for each report.
4) preprocessing/create_sentences_files.py - Create file with sentences for each report.


## Released models
- Version 0.1 - (Mar 9, 2023) Initial release
- Version 0.2 - () Model with flipped images fixed. Trained on 8 A100 GPUs in about three days.

TODO LIST:
- [ ] Add original implementation of adversarial training.
- [ ] Test with Microsoft's text encoder.
- [X] Add synthetic sentences based on other sources of information
- [ ] Add warmup time for the diffusion model
- [ ] Improve lr schedulers
- [ ] Use EMA in the diffusion model training
- [ ] Include images from other datasets, e.g. [ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345)
