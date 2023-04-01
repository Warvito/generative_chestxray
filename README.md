# Latent Diffusion Models for Chest X-Ray Generation using MONAI Generative Models

Script to train a Latent Diffusion Model based on [Pinaya et al. "Brain imaging generation with latent diffusion models.
"](https://arxiv.org/abs/2209.07162) on the MIMIC-CXR dataset using [MONAI Generative Models
](https://github.com/Project-MONAI/GenerativeModels) package.


## Instructions
### Preprocessing
After downloading the JPG images from [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) and the
associated free-text reports from [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.0.0/), you need to
preprocess the data. The following is the list of execution for preprocessing:

1) `src/python/preprocessing/organise.py` - Resizes dataset to 512 pixels in the smaller dimension
2) `src/python/preprocessing/create_ids.py` - Create files with datalist for training, validation and test using only "PA" views
3) `src/python/preprocessing/create_section_files.py` - Create file with text sections for each report.
4) `src/python/preprocessing/create_sentences_files.py` - Create file with sentences for each report.

### Training
After preprocessing, you can train the model using similar commands as in the following files (note: This project was
executed on a cluster with RunAI platform):

1) `cluster/runai/training/stage1.sh`
2) `cluster/runai/training/ldm.sh`

These files indicates which parameters and configuration file was used for training, as well how the host directories
were mounted in the used Docker container.

### Inference and evaluation
Finally, we converted the mlflow model to .pth files (for easly loading in MONAI), sampled images from the diffusion
model, and evaluated the model. The following is the list of execution for inference and evaluation:

1) `src/python/testing/convert_mlflow_to_pytorch.py` - Convert mlflow model to .pth files
2) `src/python/testing/sample_images.py` - Sample images from the diffusion model
3) `src/python/testing/compute_fid.py` - Compute FID score between generated images and real images
4) `src/python/testing/compute_msssim.py` - Measure the mean structural similarity index between images in
order to measure the diversity between them, as well between real and reconstructed images (created by the AutoencoderKL
).


## Released models
- Version 0.1 - (Mar 9, 2023) Initial release
- Version 0.2 - () Model with flipped images fixed.
