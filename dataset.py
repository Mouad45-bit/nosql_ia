from datasets import load_dataset
import pandas as pd
import ast

ds = load_dataset("PedroCJardim/QASports", "all")
train = ds["train"]

rows = []
skipped = 0

for item in train:
    raw_answer = item["answer"] # type: ignore
    #
    try:
        #
        ans = ast.literal_eval(str(raw_answer))
    except Exception:
        skipped += 1
        continue
    #
    if not isinstance(ans, dict):
        skipped += 1
        continue
    text = ans.get("text")
    #
    if not text:
        skipped += 1
        continue
    rows.append({
        "question": item["question"], # type: ignore
        "context": item["context"], # type: ignore
        "response": text,
    })

print("Exemples gardés :", len(rows))
print("Exemples ignorés :", skipped)

df = pd.DataFrame(rows)
df.to_csv("qa_sports.csv", index=False)
