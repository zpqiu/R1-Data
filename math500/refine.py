from datasets import load_dataset

ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

# ds = ds.filter(lambda x: "\\frac" in x["answer"] and (x["answer"].count("{") != 2 or x["answer"].count("}") != 2))
# ds = ds.to_pandas()
# print(ds[['problem', 'answer']].values)

correct_dict = {
    "\\frac43": "\\frac{4}{3}",
    "\\frac65": "\\frac{6}{5}",
    "\\frac 59": "\\frac{5}{9}",
    "\\frac9{19}": "\\frac{9}{19}",
    "\\frac14": "\\frac{1}{4}",
    "\\frac 34": "\\frac{3}{4}",
}

# Correct wrong answer
ds = ds.map(lambda x: {"answer": correct_dict[x["answer"]] if x["answer"] in correct_dict else x["answer"]})

ds.push_to_hub("pe-nlp/MATH-500")