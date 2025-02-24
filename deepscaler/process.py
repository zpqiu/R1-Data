import datasets

DEEPSCALER_DATASET_LIST = [
    "pe-nlp/DeepScaleR-2k-Prompt-Filtered-difficulty",
    "pe-nlp/DeepScaleR-2k-4k-Prompt-Filtered-difficulty",
    "pe-nlp/DeepScaleR-4k-8k-Prompt-Filtered-difficulty",
    "pe-nlp/DeepScaleR-8k-16k-Prompt-Filtered-difficulty",
]

dataset_list = [datasets.load_dataset(dataset_name, split="train") for dataset_name in DEEPSCALER_DATASET_LIST]

dataset = datasets.concatenate_datasets(dataset_list)
print(len(dataset))

# -1 is not a valid difficulty 
# filter out too easy problems
dataset = dataset.filter(lambda x: x["difficulty"] >= 0 and x["difficulty"] < 1)
print(len(dataset))

df = dataset.to_pandas()
df['difficulty_level'] = df['difficulty'].apply(lambda x: int(x * 10))
print(df['difficulty_level'].value_counts())

def map_to_difficulty_level(pass_rate: float) -> str:
    if pass_rate >= 0.65:
        return "easy"
    elif pass_rate <= 0.15:
        return "hard"
    else:
        return "medium"

# filter too easy problems
df = df[df['difficulty'] < 1]
df['difficulty_level'] = df['difficulty'].apply(map_to_difficulty_level)
print(df['difficulty_level'].value_counts())

dataset = dataset.map(lambda x: {**x, "difficulty_level": map_to_difficulty_level(x["difficulty"])})
dataset.push_to_hub("pe-nlp/DeepScaleR-Filtered-difficulty")
