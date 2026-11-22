import os
import re
import pandas as pd
import numpy as np

# Load source PROMISE dataset
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data")
promise_file = os.path.join(data_dir, "PROMISE_classic_nfr.csv")

df = pd.read_csv(promise_file)

# Extract project ID (number at beginning of text)
df['Project'] = df['RequirementText'].str.extract(r'^(\d+),')[0].astype(int)
df['RequirementTextClean'] = df['RequirementText'].apply(lambda x: re.sub(r'^\d+,[\'\"\s]*', '', str(x)).rstrip('\'\"'))

# Functional requirements pool for context association
fr_df = df[df['Class'] == 'F'].copy().reset_index(drop=True)

# Quality target classes
target_classes = ['PE', 'SE', 'US', 'A']
nfr_df = df[df['Class'].isin(target_classes)].copy().reset_index(drop=True)

print(f"Total Requirements: {len(df)}")
print(f"FR pool size: {len(fr_df)}")
print(f"NFR target size: {len(nfr_df)}")
print("Class breakdown:\n", nfr_df['Class'].value_counts())

# Associate functional context (by project if available, else round-robin within project)
context_list = []
for idx, row in nfr_df.iterrows():
    proj = row['Project']
    proj_frs = fr_df[fr_df['Project'] == proj]
    if len(proj_frs) > 0:
        # select deterministic matching FR in same project
        ctx = proj_frs.iloc[idx % len(proj_frs)]['RequirementTextClean']
    else:
        ctx = fr_df.iloc[idx % len(fr_df)]['RequirementTextClean']
    context_list.append(ctx)

nfr_df['FR_Context'] = context_list

# Split into P1-P10 (Train) and P11-P15 (Test)
train_df = nfr_df[nfr_df['Project'].between(1, 10)].copy().reset_index(drop=True)
test_df = nfr_df[nfr_df['Project'].between(11, 15)].copy().reset_index(drop=True)

print(f"\nTrain set (Projects P1-P10): {len(train_df)} samples")
print(train_df['Class'].value_counts().to_dict())
print(f"\nTest set (Projects P11-P15): {len(test_df)} samples")
print(test_df['Class'].value_counts().to_dict())

# Save clean split CSVs
train_df.to_csv(os.path.join(data_dir, "train_p1_p10.csv"), index=False)
test_df.to_csv(os.path.join(data_dir, "test_p11_p15.csv"), index=False)
nfr_df.to_csv(os.path.join(data_dir, "nfr_all_with_context.csv"), index=False)

print("\nSuccessfully prepared and saved clean train/test splits!")
