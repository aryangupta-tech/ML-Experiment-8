import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
df = pd.read_csv("Iris.csv")
df = df.drop("Id" , axis = 1)
x = df.drop("Species" , axis = 1)
y = df["Species"]
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)
IrisPredModel = LogisticRegression()
IrisPredModel.fit(x_train,y_train)
pred = IrisPredModel.predict(x_test)
print("Prediction of First Sample :" , pred[0])
accuracy = accuracy_score(pred,y_test)
print("Accuracy Score is :" , accuracy)

with open("iris_model.pkl" , "wb") as file:
    pickle.dump(IrisPredModel , file)
    print("Model Saved")