import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

model = joblib.load("model.pkl")
df = pd.read_csv("Iris.csv")
df = df.drop("Id", axis=1)
x = df.drop("Species", axis=1)
y = df["Species"]
pred = model.predict(x)
accuracy = accuracy_score(y, pred)
print(f"Accuracy: {accuracy}")
assert accuracy >= 0.96
assert len(pred) == len(y)
print("All test case passed")