MLOps Project – California Housing Price Prediction

This project walks you through building a complete machine learning pipeline using:

- Google Colab – to train the model
- MLflow – to track the model
- FastAPI – to serve the model
- Docker – to package and run the model as an API

The dataset used is the California Housing dataset, and the model is a simple Linear Regression.

---

Folder Structure

When you finish, your project folder will look like this:

mlops_california_housing/
├── model/ ← MLflow exported model folder
├── main.py ← FastAPI app
├── Dockerfile ← Docker instructions
└── README.md ← This file

```

---

Part 1: Train and Track Model in Google Colab

Step 1: Open Google Colab

1. Go to: [https://colab.research.google.com](https://colab.research.google.com)
2. Create a new notebook.

---

Step 2: Paste the following code in steps

#Install required packages

!pip install mlflow scikit-learn pandas

#Import packages and load the dataset

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train and log the model with MLflow


mlflow.set_experiment("CaliforniaHousing_LR")

with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")

    print(f"Model logged! MSE: {mse}")


# Download the model to your local system


import shutil
shutil.make_archive("model", 'zip', "mlruns")


Then, download the `model.zip` file from the left-hand side file browser in Colab.

---

## Part 2: Set Up the Project on Your Windows System

### Step 0: Extract the model

1. Move the downloaded `model.zip` to a folder on your computer.
2. Right-click the ZIP → Choose "Extract All".
3. Rename the extracted folder to `model`.
4. Make sure it contains files like `conda.yaml`, `MLmodel`, and `model.pkl`.

---

### Step 1: Install Docker

Docker lets you run the API without needing to install Python or other packages yourself.

1. Download Docker Desktop: [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
2. Install it and restart your system if needed.
3. Open Docker Desktop and make sure it is running.

---

## Part 3: Set Up FastAPI in VS Code

### Step 1: Install VS Code

1. Download from [https://code.visualstudio.com](https://code.visualstudio.com)
2. Install and open it.

---

### Step 2: Create a Project Folder

1. Create a folder (for example: `mlops_california_housing`) anywhere on your computer.
2. Open that folder in VS Code.
3. Inside the folder, place the following:

   * The extracted `model/` folder from Colab
   * A new file named `main.py`
   * A new file named `Dockerfile`

---

### Step 3: Add the following content to each file

#### main.py


from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn

model = mlflow.sklearn.load_model("model")

app = FastAPI()

class InputData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.post("/predict")
def predict(data: InputData):
    input_data = [[
        data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms,
        data.Population, data.AveOccup, data.Latitude, data.Longitude
    ]]
    prediction = model.predict(input_data)[0]
    return {"predicted_price": prediction}

#### Dockerfile


FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip install fastapi uvicorn mlflow scikit-learn pandas pydantic

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

Part 4: Build and Run the API

Step 1: Open Terminal

In VS Code:

- Click on `Terminal` → `New Terminal`
- Make sure the terminal is open at the project folder path

---

Step 2: Build and Run the Docker container

Run these commands one after the other:

docker build -t housing-api .
docker run -p 8000:8000 housing-api

This will start your FastAPI server on port 8000.

---

Part 5: Test the API

Step 1: Open the Swagger UI

Go to your browser and visit:

```
http://localhost:8000/docs
```

### Step 2: Send a Prediction Request

1. Click on the `/predict` section → “Try it out”
2. Paste this input:

```json
{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.984127,
  "AveBedrms": 1.02381,
  "Population": 322.0,
  "AveOccup": 2.555556,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

3. Click “Execute”.
   You’ll get a predicted house price in the response.

---
