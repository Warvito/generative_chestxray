"""Script to create json files with sentences from reports.

In order to run this script, it is necessary to run `python -m spacy download en_core_web_sm` first.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import spacy
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sectioned_file", help="Path to mimic_cxr_sectioned.csv created by create_sectioned_reports.py"
    )
    parser.add_argument(
        "--output_dir", default="./derivatives/report_sentences", help="Path to directory to save text files"
    )
    parser.add_argument("--training_ids", help="Location of file with training ids.")
    parser.add_argument("--validation_ids", help="Location of file with validation ids.")
    parser.add_argument("--test_ids", help="Location of file with test ids.")

    args = parser.parse_args()
    return args


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    section_df = pd.read_csv(args.sectioned_file, sep=",")
    nlp = spacy.load("en_core_web_sm")

    train_df = pd.read_csv(args.training_ids, sep="\t")
    val_df = pd.read_csv(args.validation_ids, sep="\t")
    test_df = pd.read_csv(args.test_ids, sep="\t")

    ids_df = pd.concat([train_df, val_df, test_df], axis=0)

    for index, row in tqdm(ids_df.iterrows(), total=ids_df.shape[0]):
        selected_reports = section_df[section_df["study"] == f"s{str(row['study_id'])}"]
        output_filename = output_dir / f"s{row['study_id']}.json"

        if len(selected_reports) == 0:
            data_dict = {"sentences": [""]}

        elif len(selected_reports) == 1:
            selected_report = selected_reports.iloc[0]
            list_of_sentences = []
            if not isinstance(selected_report["findings"], float):
                section_text = selected_report["findings"]
                section_text = re.sub("\n", "", section_text)
                section_text = re.sub(" +", " ", section_text)
                doc = nlp(section_text)
                for sent in doc.sents:
                    if len(sent.text) > 2:
                        list_of_sentences.append(sent.text)

            if not isinstance(selected_report["impression"], float):
                section_text = selected_report["impression"]
                section_text = re.sub(r"\n", "", section_text)
                section_text = re.sub(r"\d+\.", "", section_text)
                section_text = re.sub(" +", " ", section_text)
                doc = nlp(section_text)
                for sent in doc.sents:
                    if len(sent.text) > 2:
                        list_of_sentences.append(sent.text)

            # save dict as json file
            if len(list_of_sentences) > 0:
                data_dict = {"sentences": list_of_sentences}

            else:
                data_dict = {"sentences": [""]}

        with open(output_filename, "w") as f:
            json.dump(data_dict, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
