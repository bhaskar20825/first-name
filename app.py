# app.py
import sys
from fastapi import FastAPI, HTTPException
from transformers import pipeline
import uvicorn

# Create the FastAPI app
app = FastAPI(title="Hugging Face Sentiment Analysis API")

# Load the Hugging Face model for sentiment analysis
model_name = "vishodi/First-Name-Classification"
classifier = pipeline("sentiment-analysis", model=model_name)

# Define the /predict endpoint
@app.get("/predict")
def predict(text: str):
    try:
        result = classifier(text)
        return result[0]  # Returns something like {'label': 'POSITIVE', 'score': 0.99}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Test function that uses TestClient (imported only here)
def run_tests():
    from fastapi.testclient import TestClient  # Imported here so it won't affect production
    client = TestClient(app)
    test_text = "I love AI!"
    response = client.get("/predict", params={"text": test_text})
    print("Test response:", response.json())

if __name__ == "__main__":
    # If you run the script with the argument 'test', execute the test function
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        # Otherwise, start the server; Render will run this command.
        uvicorn.run("app:app", host="0.0.0.0", port=8000)
