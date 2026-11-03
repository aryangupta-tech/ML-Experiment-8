#import python version
FROM python:3.14.7

#make a working directory
WORKDIR /app

#copy libraries from requirements.txt
COPY requirements.txt .

#for installing all the libraries from file , if want to keep dump file remove --no-cache-dir , -r used to read requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

#import pickle file
COPY iris_model.pkl .

#import app.py file
COPY app.py .

#to run dockerfile finally 
CMD ["python" , "app.py"]