import json
import re
from pathlib import Path

import pandas as pd
import spacy
from tqdm import tqdm

output_dir = Path("/media/walter/Storage/DATASETS/MIMIC/mimic/derivatives/report_sentences")
output_dir.mkdir(exist_ok=True)

section_df = pd.read_csv(
    "/media/walter/Storage/Projects/generative_mimic/outputs/reports/mimic_cxr_sectioned.csv", sep=","
)
nlp = spacy.load("en_core_web_sm")

train_df = pd.read_csv("/media/walter/Storage/Projects/generative_mimic/outputs/ids/train.tsv", sep="\t")
val_df = pd.read_csv("/media/walter/Storage/Projects/generative_mimic/outputs/ids/validation.tsv", sep="\t")
test_df = pd.read_csv("/media/walter/Storage/Projects/generative_mimic/outputs/ids/test.tsv", sep="\t")

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
