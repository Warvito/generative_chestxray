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

1) `cluster/runai/training/stage1.sh` - Command to start to execute in the server the training the first stage of the model.
The main python script in for this is the `src/python/training/train_aekl.py` script. The `--volume` flags indicate how the dataset
is mounted in the Docker container.
2) `src/python/training/eda_ldm_scaling_factor.py` - Script to find the best scaling factor for the latent diffusion model.
3) `cluster/runai/training/ldm.sh` - Command to start to execute in the server the training the diffusion model on the latent representation.
The main python script in for this is the `src/python/training/train_ldm.py` script. The `--volume` flags indicate how the dataset
is mounted in the Docker container.

These `.sh` files indicates which parameters and configuration file was used for training, as well how the host directories
were mounted in the used Docker container.

### Inference and evaluation
Finally, we converted the mlflow model to .pth files (for easly loading in MONAI), sampled images from the diffusion
model, and evaluated the model. The following is the list of execution for inference and evaluation:

1) `src/python/testing/convert_mlflow_to_pytorch.py` - Convert mlflow model to .pth files
2) `src/python/testing/sample_images.py` - Sample images from the diffusion model. `cluster/runai/testing/sampling_unconditioned.sh` shows
how to execute this script in the server to generate the 1000 samples used in the following scripts.
3) `src/python/testing/compute_msssim_reconstruction.py` - Measure the mean structural similarity index between images and
reconstruction to measure the preformance of the first stage.
4) `src/python/testing/compute_msssim_sample.py` - Measure the mean structural similarity index between test images and
samples in order to measure the diversity of the synthetic data.
5) `src/python/testing/compute_msssim_test_set.py` - Measure the mean structural similarity index between test images
to measure the diversity of the reference test set.
6) `src/python/testing/compute_fid.py` - Compute FID score between generated images and real images.

## Released models
- Version 0.1 - (Mar 9, 2023) Initial release
- Version 0.2 - () Model with flipped images fixed.
