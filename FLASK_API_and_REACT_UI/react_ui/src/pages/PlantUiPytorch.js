import React, { useState } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';

function PlantUiPytorch() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file)); // Generate a preview URL for the image

    // Automatically trigger prediction after file selection
    const formData = new FormData();
    formData.append('file', file);

    try {
      setLoading(true);
      const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.error) {
        setError(response.data.error);
        setPrediction(null);
      } else {
        // Update with correct field names ('label' and 'confidence')
        setPrediction({
          label: response.data.label,  // Display predicted label
          confidence: response.data.confidence,  // Display confidence
        });
        setError(null);
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      setError('An error occurred while processing the image or server is not running.');
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  const triggerFileSelect = () => {
    document.getElementById('fileUploadInput').click();
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      minHeight: '100vh',
    }}>
      {/* Header */}
      <div style={{ background: 'blue', border: '1px solid #ccc', marginBottom: '20px' }}>
        <h1 style={{ textAlign: 'center', color: 'white' }}>
          Potato Plant Disease Prediction with Artificial Intelligence (Pytorch)
        </h1>
      </div>

      {/* Main content */}
      <div style={{ textAlign: 'center', flex: '1' }}>
        {/* Hidden file input */}
        <input
          id="fileUploadInput"
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />

        {/* Custom upload button with conditional styling */}
        <button
          onClick={triggerFileSelect}
          style={{
            marginBottom: '20px',
            padding: '10px 20px',
            backgroundColor: selectedFile ? '#FFC107' : '#007BFF', // Warning color for "Change" and primary color for "Upload"
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
          }}
        >
          {selectedFile ? 'Change Plant Leaf Image' : 'Upload Plant Leaf Image'}
        </button>
        <br />
        {previewUrl && (
          <div
            style={{
              border: '1px solid #ccc',
              borderRadius: '8px',
              padding: '20px',
              display: 'inline-block',
              textAlign: 'center',
              maxWidth: '300px',
            }}
          >
            <img
              src={previewUrl}
              alt="Uploaded"
              style={{ width: '100%', borderRadius: '8px', marginBottom: '10px' }}
            />
            {loading ? (
              <p style={{ fontSize: '16px', color: '#666' }}>Predicting...</p>
            ) : error ? (
              <p style={{ fontSize: '16px', color: 'red' }}>{error}</p>
            ) : prediction ? (
              <div>
                <span style={{ color: 'blue', fontSize: '20px' }}>Predicted Results</span>
                <h2 style={{ fontSize: '18px' }}>Label: {prediction.label}</h2>
                <p style={{ fontSize: '16px', color: '#666' }}>
                  Confidence: {(prediction.confidence * 100).toFixed(2)}%
                </p>
              </div>
            ) : (
              <p style={{ fontSize: '16px', color: '#666' }}>No prediction yet</p>
            )}
          </div>
        )}
      </div>

      {/* Footer */}
      <div style={{
        background: 'blue',
        borderTop: '1px solid #ccc',
        padding: '20px 0',
        textAlign: 'center',
        color: 'white',
      }}>
        <p>@Etienne AI Engineer 2024</p>
        <a href='https://youtu.be/DQFerxe6o5g?si=fUuLa-XitleZzrvH'>Documentation</a> <br />
        <Link to="/" style={{ color: 'white' }}>Tensorflow</Link>
      </div>
    </div>
  );
}

export default PlantUiPytorch;
