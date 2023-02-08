""" Script to organise and downsampling the images from the dataset.

During resizing, the smaller edge of the image will be matched to this number.
"""
from pathlib import Path
from torchvision.transforms import Resize
from torchvision.datasets.folder import default_loader
import zipfile


def main():
    source_dir = Path('/sourcedata/')
    raw_dir = Path('/rawdata/')

    resize_fn = Resize(512)

    for image_file in source_dir.glob('**/*.jpg'):
        image = default_loader(image_file)
        new_image = resize_fn(image)

        new_dir = raw_dir / image_file.relative_to("/sourcedata/").parent
        new_dir.mkdir(parents=True, exist_ok=True)
        new_image.save(new_dir / image_file.name)

    # Unzip medical reports into raw_dir
    with zipfile.ZipFile(source_dir / "mimic-cxr-reports.zip", 'r') as zip_ref:
        zip_ref.extractall(raw_dir)


if __name__ == '__main__':
    main()