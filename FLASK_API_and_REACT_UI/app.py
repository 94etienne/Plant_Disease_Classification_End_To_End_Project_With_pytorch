from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import torch
from model import CNNModel  # Import the CNNModel class from model.py
import torchvision.transforms as transforms
from PIL import Image
import io

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Initialize the model and load the weights
model = CNNModel(n_classes=3)  # Adjust n_classes if needed (e.g., 3 for 3 classes)
model.load_state_dict(torch.load("../version_2_on_CPU/model_1.pth"))  # Make sure the path is correct
model.eval()  # Set the model to evaluation mode

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Ensure the image is resized to 256x256
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# API endpoint to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # List of label names (adjust these to match your actual labels)
        label_names = ['Early Blight', 'Late Blight', 'Healthy']

        # Check if the file is in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        img_file = request.files['file']

        # Check if the file has a valid extension
        if img_file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Open the image file
        img = Image.open(io.BytesIO(img_file.read()))
        img = transform(img).unsqueeze(0)  # Apply transformations and add batch dimension

        # Predict the class
        with torch.no_grad():
            output = model(img)
            _, predicted_class = torch.max(output, 1)
            confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_class].item()

        # Get the label name corresponding to the predicted class index
        predicted_label = label_names[int(predicted_class.item())]

        return jsonify({
            'label': predicted_label,  # Send label name instead of class index
            'confidence': confidence
        })

    except Exception as e:
        # Log the exception and return error
        app.logger.error(f"Error processing image: {e}")
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    app.run(debug=True)
