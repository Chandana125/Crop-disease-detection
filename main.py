import os
from fastapi.middleware.cors import CORSMiddleware
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from starlette.responses import JSONResponse
import uvicorn
import logging

logging.basicConfig(filename="app.log", level=logging.DEBUG)

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8081",
    "http://127.0.0.1:5500"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Specify the path to the directory containing the SavedModel
model_directory = r"C:\Users\Chandana\Desktop\CropHealthMonitoring\saved models-20231024T052323Z-001"

# Check if the directory exists
if os.path.exists(model_directory):
    print("SavedModel directory exists.")
else:
    print("SavedModel directory does not exist.")

# Load your TensorFlow model
saved_model_path = r"C:\Users\Chandana\Desktop\CropHealthMonitoring\saved models-20231024T052323Z-001\saved models\1"
MODEL = None  # Initialize the MODEL variable

# Load the model from the specified directory
model = tf.keras.models.load_model(saved_model_path)

# Verify that the model is loaded successfully
model.summary()

try:
    MODEL = tf.keras.models.load_model(saved_model_path)
except OSError as e:
    print(f"An error occurred: {e}")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
# Define the directory where uploaded files will be saved
upload_dir = "uploads"

# Create the directory if it doesn't exist
os.makedirs(upload_dir, exist_ok=True)

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

def read_file_as_image(data) -> np.ndarray:
    try:
        # Attempt to open the image data
        dataBytesIO = io.BytesIO(data)
        image = Image.open(dataBytesIO)
        return np.array(image)
    except Exception as e:
        print(f"Error: {e}")
        return None
VALID_IMAGE_EXTENSIONS = {".jpg",".jpeg",".png"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Access the global MODEL variable
    global MODEL

    if MODEL is None:
        return JSONResponse(content={"error": "Model not loaded"})
    if not file.filename.lower().endswith(tuple(VALID_IMAGE_EXTENSIONS)):
        return JSONResponse(content={"error": "Invalid image format"})

    # Read the uploaded image and convert it to a NumPy array
    image = read_file_as_image(await file.read())

    if image is None:
        return JSONResponse(content={"error": "Invalid image format"})

    # Perform the prediction using the loaded model
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)

    # Assuming your model outputs class probabilities, get the class with the highest probability
    predicted_class_index = np.argmax(predictions[0])
    confidence=np.max(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_index]
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence}")

    

    return JSONResponse(content={"message": "Prediction complete", "predicted_class": predicted_class,
                            "confidence": float(confidence)})

           
    

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8081)
