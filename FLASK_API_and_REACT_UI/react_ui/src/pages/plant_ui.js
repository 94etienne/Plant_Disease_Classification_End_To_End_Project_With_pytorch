import React, { useState } from 'react';
import axios from 'axios';

function PlantUi() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file)); // Generate a preview URL for the image
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      setLoading(true);
      const response = await axios.post('http://localhost:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setPrediction(response.data);
    } catch (error) {
      console.error('Error uploading file:', error);
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ textAlign: 'center' }}>
      <h1>Image Prediction</h1>
      <form onSubmit={handleSubmit} style={{ marginBottom: '20px' }}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button type="submit" disabled={!selectedFile || loading} style={{ marginTop: '10px' }}>
          {loading ? 'Predicting...' : 'Upload & Predict'}
        </button>
      </form>

      {previewUrl && (
        <div style={{ border: '1px solid #ccc', borderRadius: '8px', padding: '20px', display: 'inline-block', textAlign: 'center', maxWidth: '300px' }}>
          <img
            src={previewUrl}
            alt="Uploaded"
            style={{ width: '100%', borderRadius: '8px', marginBottom: '10px' }}
          />
          {prediction ? (
            <div>
              <h2 style={{ fontSize: '18px' }}>Label: {prediction.class}</h2>
              <p style={{ fontSize: '16px', color: '#666' }}>Confidence: {(prediction.confidence * 100).toFixed(2)}%</p>
            </div>
          ) : (
            <p style={{ fontSize: '16px', color: '#666' }}>No prediction yet</p>
          )}
        </div>
      )}
    </div>
  );
}

export default PlantUi;
