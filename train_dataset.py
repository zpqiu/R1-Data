import datasets

MATH_EPISODES = 2

DIFFICULTY_LEVEL_EPISODES = {
    'easy': 2,
    'medium': 15,
    'hard': 20,
}

# 加载原始数据集
math_dataset = datasets.load_dataset("pe-nlp/math-level3to5-Filtered", split="train")
math_dataset = math_dataset.map(lambda x: {**x, "problem": x['question']})
math_dataset = math_dataset.select_columns(["problem", "ground_truth_answer"])


deepscaler_dataset = datasets.load_dataset("pe-nlp/DeepScaleR-Filtered-difficulty", split="train")
# 创建deepscaler_dataset的副本
easy_dataset = deepscaler_dataset.filter(lambda x: x['difficulty_level'] == 'easy').select_columns(["problem", "ground_truth_answer"])
medium_dataset = deepscaler_dataset.filter(lambda x: x['difficulty_level'] == 'medium').select_columns(["problem", "ground_truth_answer"])
hard_dataset = deepscaler_dataset.filter(lambda x: x['difficulty_level'] == 'hard').select_columns(["problem", "ground_truth_answer"])


dataset_list = []

for i in range(MATH_EPISODES):
    ds_copy = math_dataset.select(range(len(math_dataset))).shuffle(seed=42 + i)
    dataset_list.append(ds_copy)

for i in range(DIFFICULTY_LEVEL_EPISODES['easy']):
    ds_copy = easy_dataset.select(range(len(easy_dataset))).shuffle(seed=100 + i)
    dataset_list.append(ds_copy)

for i in range(DIFFICULTY_LEVEL_EPISODES['medium']):
    ds_copy = medium_dataset.select(range(len(medium_dataset))).shuffle(seed=200 + i)
    dataset_list.append(ds_copy)

for i in range(DIFFICULTY_LEVEL_EPISODES['hard']):
    ds_copy = hard_dataset.select(range(len(hard_dataset))).shuffle(seed=300 + i)
    dataset_list.append(ds_copy)

train_dataset = datasets.concatenate_datasets(dataset_list)
print(len(train_dataset))
train_dataset.push_to_hub("pe-nlp/math-cl")
