# Vertex AI: Training, Serving, and Monitoring a Custom Model — California Housing Price Prediction

## Overview
This project demonstrates the end-to-end workflow of training, deploying, and monitoring a custom TensorFlow model on Google Cloud Vertex AI using a custom Docker container. Instead of the default Auto MPG dataset, this project uses the **California Housing dataset** to predict median house prices based on features like median income, house age, average rooms, population, and location.

## Dataset
The [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) from scikit-learn contains 20,640 samples with 8 features:

| Feature | Description |
|---------|-------------|
| MedInc | Median income in block group |
| HouseAge | Median house age in block group |
| AveRooms | Average number of rooms per household |
| AveBedrms | Average number of bedrooms per household |
| Population | Block group population |
| AveOccup | Average number of household members |
| Latitude | Block group latitude |
| Longitude | Block group longitude |

**Target:** Median house value (in $100,000s)

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Training    │     │   Vertex AI  │     │   Model      │     │  Endpoint    │
│  Code +      │────▶│   Training   │────▶│   Registry   │────▶│  Deployment  │
│  Dockerfile  │     │   Job        │     │              │     │  + Monitoring│
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

## Model
A Sequential neural network built with TensorFlow/Keras:
- Input layer: 8 features (normalized)
- Hidden layer 1: 64 units, ReLU activation
- Hidden layer 2: 64 units, ReLU activation
- Output layer: 1 unit (predicted price)
- Optimizer: RMSprop (lr=0.001)
- Loss: Mean Squared Error
- Early stopping with patience=10 on validation loss

**Results:** Test MAE: 0.37, Test MSE: 0.28

## Project Structure
```
├── Dockerfile
├── trainer/
│   └── train.py
├── Images/
│   ├── 0.png - 26.png
└── README.md
```

## Steps to Reproduce

### Task 1: Set Up Your Environment
1. Create a GCP project and enable the following APIs:
   - Compute Engine API
   - Vertex AI API
   - Container Registry API
   - Artifact Registry API

![Set up](./Images/0.png)

2. Create a Vertex AI Workbench Instance

![Workbench Instance](./Images/2.png)

3. Create a Cloud Storage bucket:
   ```bash
   PROJECT_ID=$(gcloud config get-value project)
   BUCKET_NAME="gs://${PROJECT_ID}-bucket"
   gsutil mb -l us-central1 $BUCKET_NAME
   ```



### Task 2: Containerize Training Code
1. Create the project structure:
   ```bash
   mkdir -p mpg/trainer
   cd mpg
   ```
2. Create `trainer/train.py` with the California Housing training code
3. Create the `Dockerfile`:
   ```dockerfile
   FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-3
   WORKDIR /
   COPY trainer /trainer
   ENTRYPOINT ["python", "-m", "trainer.train"]
   ```
4. Build, test, and push the container:
   ```bash
   PROJECT_ID=$(gcloud config get-value project)
   IMAGE_URI="gcr.io/$PROJECT_ID/california-housing:v1"
   docker build ./ -t $IMAGE_URI
   docker run $IMAGE_URI
   docker push $IMAGE_URI
   ```

![Jupyter Lab](./Images/3.png)
![Jupyter Lab](./Images/4.png)
![Jupyter Lab](./Images/5.png)
![Jupyter Lab](./Images/8.png)

### Task 3: Run a Training Job on Vertex AI
1. Navigate to Vertex AI → Training → Create
2. Select Custom Container and provide the container image URI
3. Configure compute (n1-standard-4, no GPU)
4. Start the training job

![Training](./Images/9.png)
![Training](./Images/11.png)
![Training](./Images/12.png)
![Training](./Images/14.png)

### Task 4: Deploy Model to Endpoint
1. Import the trained model into the Model Registry
   - Model artifact location: `gs://<BUCKET>/california_housing/model`
   - Framework: TensorFlow 2.3
2. Deploy the model to an endpoint
3. Test predictions with sample input:
   ```json
   {
     "instances": [
       [1.2735, 37.0, 5.0039, 1.0106, 959.0, 2.4158, 37.79, -122.26]
     ]
   }
   ```

![Deployment](./Images/17.png)
![Deployment](./Images/19.png)
![Deployment](./Images/20.png)


### Task 5: Model Monitoring
1. Navigate to Vertex AI → Monitoring → Configure Monitoring
2. Select the deployed model and define the schema (8 input features + Price output)
3. Set training data source for skew detection
4. Configure monitoring objectives for input feature drift and inference output drift

![Monitoring](./Images/21.png)
![Monitoring](./Images/23.png)
![Monitoring](./Images/25.png)


## Key Modifications from Original Lab
- **Dataset:** California Housing (20,640 samples, 8 features) instead of Auto MPG (398 samples, 9 features)
- **Target variable:** Median house price instead of miles per gallon
- **Larger dataset:** ~50x more training samples for better model generalization
- **Different feature engineering:** Normalization applied to housing-specific features (income, location, population)

## Technologies Used
- Google Cloud Platform (Vertex AI, Cloud Storage, Container Registry)
- TensorFlow 2.3 / Keras
- Docker
- scikit-learn (for dataset loading)
- Python 3.7

## Conclusion
Successfully trained and deployed a TensorFlow model using Vertex AI with a custom container on the California Housing dataset. The model predicts median house prices and is served through a Vertex AI endpoint with monitoring configured for training-serving skew detection.
