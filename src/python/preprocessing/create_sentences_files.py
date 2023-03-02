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

for index, row in tqdm(section_df.iterrows(), total=section_df.shape[0]):
    list_of_sentences = []
    if not isinstance(row["findings"], float):
        section_text = row["findings"]
        section_text = re.sub("\n", "", section_text)
        section_text = re.sub(" +", " ", section_text)
        doc = nlp(section_text)
        for sent in doc.sents:
            if len(sent.text) > 2:
                list_of_sentences.append(sent.text)

    if not isinstance(row["impression"], float):
        section_text = row["impression"]
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
        filename = output_dir / f"{row['study']}.json"
        with open(filename, "w") as f:
            json.dump(data_dict, f, indent=4)
