""" Script to create train, validation and test data lists with paths to images and radiological reports. """
import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata_path", help="Path to mimic-cxr-2.0.0-metadata.csv file.")
    parser.add_argument("--output_dir", help="Path to directory to save files with paths.")

    args = parser.parse_args()
    return args


def main(args):
    # read the data
    metadata_df = pd.read_csv(args.metadata_path, na_values="-")

    # filter the data to use only "PA" view in ViewPosition column
    metadata_df = metadata_df[metadata_df["ViewPosition"] == "PA"]

    # create data list of paths to the images and radiological reports
    data_list = []
    for index, row in metadata_df.iterrows():
        data_list.append(
            {
                "image": f"/data/rawdata/files/p{str(int(row['subject_id']))[:2]}/p{str(int(row['subject_id']))}/"
                f"s{int(row['study_id'])}/{row['dicom_id']}.jpg",
                "report": f"/data/rawdata/files/p{str(int(row['subject_id']))[:2]}/p{str(int(row['subject_id']))}/"
                f"s{int(row['study_id'])}.txt",
            }
        )

    # transform data list to dataframe and shuffle
    data_df = pd.DataFrame(data_list)
    data_df = data_df.sample(frac=1).reset_index(drop=True)

    # split in train, validation and test data lists. The total number of PA images in data_df is 96161.
    train_data_list = data_df[:90000]
    val_data_list = data_df[90000:91161]
    test_data_list = data_df[91161:]

    # save the data lists
    output_dir = Path(args.output_dir)
    train_data_list.to_csv(output_dir / "train_ids.tsv", index=False, sep="\t")
    val_data_list.to_csv(output_dir / "val_ids.tsv", index=False, sep="\t")
    test_data_list.to_csv(output_dir / "test_ids.tsv", index=False, sep="\t")


if __name__ == "__main__":
    args = parse_args()
    main(args)
