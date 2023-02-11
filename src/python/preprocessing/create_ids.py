import pandas as pd

# read the data
metadata_df = pd.read_csv(
    "/media/walter/Storage/Projects/generative_mimic/outputs/mimic-cxr-2.0.0-metadata.csv", na_values="-"
)

# filter the data to use only "PA" view in ViewPosition column
metadata_df = metadata_df[metadata_df["ViewPosition"] == "PA"]

# create data list of paths to the images and radiological reports
data_list = []
for index, row in metadata_df.iterrows():
    data_list.append(
        {
            "image": f"/data/rawdata/files/p{str(int(row['subject_id']))[:2]}/p{str(int(row['subject_id']))}/s{int(row['study_id'])}/{row['dicom_id']}.jpg",
            "report": f"/data/rawdata/files/p{str(int(row['subject_id']))[:2]}/p{str(int(row['subject_id']))}/s{int(row['study_id'])}.txt",
        }
    )

# transform data list to dataframe and shuffle
data_df = pd.DataFrame(data_list)
data_df = data_df.sample(frac=1).reset_index(drop=True)

# split in train, validation and test
train_data_list = data_df[:90000]
val_data_list = data_df[90000:91161]
test_data_list = data_df[91161:]

# save the data lists
train_data_list.to_csv(
    "/media/walter/Storage/Projects/generative_mimic/outputs/ids/train_ids.tsv", index=False, sep="\t"
)
val_data_list.to_csv("/media/walter/Storage/Projects/generative_mimic/outputs/ids/val_ids.tsv", index=False, sep="\t")
test_data_list.to_csv("/media/walter/Storage/Projects/generative_mimic/outputs/ids/test_ids.tsv", index=False, sep="\t")
