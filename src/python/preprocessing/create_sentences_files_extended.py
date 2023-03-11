""" Script to create json files with sentences from reports together with synthetic sentences generated from
mimic-cxr-2.0.0-chexpert.csv.

This script extends create_sentences_files.py by generating the sentences files not just with original real sentences,
but with sentences created form aditional source of informations about the subjects. In our case, we only add
sentences when one of the CheXpert labels is positive, except when the label is "No Finding" and "Pleural Other".
"""
from __future__ import annotations

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
    parser.add_argument("--chexpert_file", help="Path to mimic-cxr-2.0.0-chexpert.csv downloaded from MIMIC-CXR-JPG")
    parser.add_argument(
        "--output_dir", default="./derivatives/report_sentences", help="Path to directory to save text files"
    )
    parser.add_argument("--training_ids", help="Location of file with training ids.")
    parser.add_argument("--validation_ids", help="Location of file with validation ids.")
    parser.add_argument("--test_ids", help="Location of file with test ids.")

    args = parser.parse_args()
    return args


def create_synthetic_sentences(entity: str, adjective: str | None = None) -> list[str]:
    list_of_sentences = []
    list_of_sentences.append(f"The patient has {entity.lower()}.")
    if entity.lower() in ["atelectasis", "edema", "enlarged cardiomediastinum"]:
        list_of_sentences.append(f"There is an {entity.lower()}.")
    else:
        list_of_sentences.append(f"There is a {entity.lower()}.")

    if adjective is not None and adjective.lower() in ["moderate", "severe", "mild"]:
        list_of_sentences.append(f"The patient has {adjective.lower()} {entity.lower()}.")
        list_of_sentences.append(f"There is an {adjective.lower()} {entity.lower()}.")

    return list_of_sentences


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    section_df = pd.read_csv(args.sectioned_file, sep=",")
    nlp = spacy.load("en_core_web_sm")

    train_df = pd.read_csv(args.training_ids, sep="\t")
    val_df = pd.read_csv(args.validation_ids, sep="\t")
    test_df = pd.read_csv(args.test_ids, sep="\t")

    chexpert_df = pd.read_csv(args.chexpert_file, sep=",")
    chexpert_columns = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
        "Support Devices",
    ]

    ids_df = pd.concat([train_df, val_df, test_df], axis=0)

    for index, row in tqdm(ids_df.iterrows(), total=ids_df.shape[0]):
        selected_reports = section_df[section_df["study"] == f"s{str(row['study_id'])}"]

        output_filename = output_dir / f"s{row['study_id']}.json"

        doc = None
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

        # Extend existing sentences with synthetic sentences using Chexpert labels
        selected_chexpert = chexpert_df[chexpert_df["study_id"] == row["study_id"]]
        for column in chexpert_columns:
            if selected_chexpert[column].iloc[0] == 1.0:
                data_dict["sentences"].extend(create_synthetic_sentences(column))

                # Check if there is an adjective modifying the entity
                if doc is not None:
                    for token in doc:
                        if token.head.text == column and token.dep_ == "amod":
                            if token.text in ["moderate", "severe", "mild", "small", "large", "big"]:
                                data_dict["sentences"].extend(create_synthetic_sentences(column, token.text))

        with open(output_filename, "w") as f:
            json.dump(data_dict, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
