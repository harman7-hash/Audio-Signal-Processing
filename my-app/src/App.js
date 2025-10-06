// frontend/src/App.js
import React, { useState, useRef } from 'react';
import './App.css';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  
  // Recording states
  const [isRecording, setIsRecording] = useState(false);
  const [recordedAudio, setRecordedAudio] = useState(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recordingIntervalRef = useRef(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setRecordedAudio(null);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (event) => {
    event.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragOver(false);
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('audio/')) {
      setSelectedFile(file);
      setRecordedAudio(null);
    }
  };

  // Recording Functions
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 44100,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        } 
      });
      
      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const audioUrl = URL.createObjectURL(audioBlob);
        setRecordedAudio({
          blob: audioBlob,
          url: audioUrl,
          name: `recording-${new Date().getTime()}.webm`
        });
        setSelectedFile(null);
      };

      mediaRecorderRef.current.start(1000); // Collect data every second
      setIsRecording(true);
      setRecordingTime(0);

      recordingIntervalRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

    } catch (error) {
      console.error('Error starting recording:', error);
      alert('Error accessing microphone. Please check permissions.');
    }
  };

// Helper: Convert AudioBuffer to WAV Blob
const audioBufferToWav = (buffer, sampleRate = 44100) => {
  const numChannels = buffer.numberOfChannels;
  const samples = buffer.getChannelData(0);
  const bufferLength = samples.length * numChannels * 2 + 44;
  const wavBuffer = new ArrayBuffer(bufferLength);
  const view = new DataView(wavBuffer);

  // RIFF header
  const writeString = (view, offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  let offset = 0;
  writeString(view, offset, 'RIFF'); offset += 4;
  view.setUint32(offset, bufferLength - 8, true); offset += 4;
  writeString(view, offset, 'WAVE'); offset += 4;
  writeString(view, offset, 'fmt '); offset += 4;
  view.setUint32(offset, 16, true); offset += 4;
  view.setUint16(offset, 1, true); offset += 2; // PCM
  view.setUint16(offset, numChannels, true); offset += 2;
  view.setUint32(offset, sampleRate, true); offset += 4;
  view.setUint32(offset, sampleRate * numChannels * 2, true); offset += 4;
  view.setUint16(offset, numChannels * 2, true); offset += 2;
  view.setUint16(offset, 16, true); offset += 2; // bits per sample
  writeString(view, offset, 'data'); offset += 4;
  view.setUint32(offset, bufferLength - offset - 4, true); offset += 4;

  // Write PCM samples
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return new Blob([view], { type: 'audio/wav' });
};

// Updated stopRecording
const stopRecording = async () => {
  if (mediaRecorderRef.current && isRecording) {
    mediaRecorderRef.current.stop();
    mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    setIsRecording(false);
    clearInterval(recordingIntervalRef.current);

    // Convert recorded audio to WAV
    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
    const arrayBuffer = await audioBlob.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    const wavBlob = audioBufferToWav(audioBuffer, audioBuffer.sampleRate);

    const audioUrl = URL.createObjectURL(wavBlob);
    setRecordedAudio({
      blob: wavBlob,       // now WAV
      url: audioUrl,
      name: `recording-${new Date().getTime()}.wav`
    });
    setSelectedFile(null);
  }
};


  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Prediction Functions
  const sendToBackend = async (audioBlob, filename) => {
    const formData = new FormData();
    formData.append('file', audioBlob, filename);

    try {
      const response = await fetch(`http://localhost:8000/api/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      return await response.json();
    } catch (error) {
      throw error;
    }
  };

  const handlePredict = async () => {
    let fileToSend = selectedFile;
    let filename = selectedFile?.name;

    if (recordedAudio) {
      fileToSend = recordedAudio.blob;
      filename = recordedAudio.name;
    }

    if (!fileToSend) {
      alert('Please select an audio file or record audio first');
      return;
    }

    setLoading(true);
    try {
      const result = await sendToBackend(fileToSend, filename);
      setPredictions([result]);
    } catch (error) {
      console.error('Error:', error);
      alert('Error making prediction: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const clearRecording = () => {
    setRecordedAudio(null);
    if (recordedAudio?.url) {
      URL.revokeObjectURL(recordedAudio.url);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
  };

  return (
    <div className="App">
      {/* Header Section */}
      <header className="app-header">
        <div className="header-content">
          <h1 className="header-title">Real-time Detection</h1>
          <p className="header-subtitle">
            Upload audio files or record live audio for anomaly detection
          </p>
        </div>
      </header>

      <div className="container">
        {/* Main Content Grid */}
        <div className="content-grid">
          {/* File Upload Section */}
          <div className="input-section">
            <div className="section-header">
              <h2>Audio Input</h2>
            </div>
            
            {/* File Upload Detection */}
            <div className="input-card">
              <h3 className="input-title">File Upload Detection</h3>
              <div
                className={`upload-area ${dragOver ? 'drag-over' : ''} ${selectedFile ? 'has-file' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                {selectedFile ? (
                  <div className="file-selected">
                    <div className="file-icon">📁</div>
                    <div className="file-info">
                      <p className="file-name">{selectedFile.name}</p>
                      <p className="file-size">
                        {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                      </p>
                    </div>
                    <button onClick={clearFile} className="clear-btn">
                      ✕
                    </button>
                  </div>
                ) : (
                  <>
                    <div className="upload-icon">📥</div>
                    <div className="upload-text">
                      <p className="upload-title">Drag & drop audio files here</p>
                      <p className="upload-subtitle">
                        Supports WAV, MP3, and other audio formats
                      </p>
                    </div>
                    <input
                      type="file"
                      accept="audio/*"
                      onChange={handleFileSelect}
                      className="file-input"
                      id="file-input"
                    />
                    <label htmlFor="file-input" className="browse-btn">
                      Browse Files
                    </label>
                  </>
                )}
              </div>
            </div>

            {/* Real-time Recording */}
            <div className="input-card">
              <h3 className="input-title">Real-time Recording</h3>
              <div className="recording-area">
                {!isRecording && !recordedAudio ? (
                  <button 
                    onClick={startRecording}
                    className="record-btn start-btn"
                  >
                    <span className="btn-icon">●</span>
                    Start Recording
                  </button>
                ) : isRecording ? (
                  <div className="recording-active">
                    <div className="recording-visualizer">
                      <div className="visualizer-bars">
                        {[...Array(8)].map((_, i) => (
                          <div key={i} className="bar" style={{ animationDelay: `${i * 0.1}s` }} />
                        ))}
                      </div>
                      <div className="recording-info">
                        <div className="recording-status">
                          <span className="recording-dot"></span>
                          Recording...
                        </div>
                        <div className="recording-timer">
                          {formatTime(recordingTime)}
                        </div>
                      </div>
                    </div>
                    <button onClick={stopRecording} className="record-btn stop-btn">
                      <span className="btn-icon">■</span>
                      Stop Recording
                    </button>
                  </div>
                ) : (
                  <div className="recording-complete">
                    <div className="recording-preview">
                      <audio controls src={recordedAudio.url} className="audio-player" />
                    </div>
                    <div className="recording-actions">
                      <button onClick={clearRecording} className="action-btn secondary">
                        Clear
                      </button>
                      <button onClick={startRecording} className="action-btn primary">
                        Record Again
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Analyze Button */}
            <button 
              onClick={handlePredict} 
              disabled={(!selectedFile && !recordedAudio) || loading}
              className="analyze-btn"
            >
              {loading ? (
                <>
                  <div className="loading-spinner"></div>
                  Analyzing...
                </>
              ) : (
                'Analyze Audio'
              )}
            </button>
          </div>

          {/* Results Section */}
          {predictions.length > 0 && (
            <div className="results-section">
              <div className="section-header">
                <h2>Detection Results</h2>
              </div>
              <div className="results-grid">
                {predictions.map((pred, index) => (
                  <div key={index} className={`result-card ${pred.error > pred.threshold ? 'anomalous' : 'normal'}`}>
                    <div className="result-header">
                      <div className="result-icon">
                        {pred.error > pred.threshold ? '🚨' : '✅'}
                      </div>
                      <div className="result-title">
                        <h3>{pred.error > pred.threshold ? 'ABNORMAL' : 'NORMAL'}</h3>
                        <span className={`status-badge ${pred.error > pred.threshold ? 'anomalous' : 'normal'}`}>
                          {pred.error > pred.threshold ? 'ABNORMAL' : 'NORMAL'}
                        </span>
                      </div>
                      {pred.threshold > pred.error && (
                        <div className="processing-time">
                          <div className="label">Processing Time</div>
                          <div className="value">{pred.processing_time_ms || 0}ms</div>
                        </div>
                      )}
                    </div>

                    <div className="result-details">
                      <div className="metric">
                        <label>Threshold</label>
                        <div className="metric-value">{pred.threshold?.toFixed(6)}</div>
                      </div>
                      <div className="metric">
                        <label>Error</label>
                        <div className="metric-value">{pred.error?.toFixed(6)}</div>
                      </div>
                    </div>

                    <div className="feature-grid">
                      <div className="feature-card">
                        <div className="feature-label">Spectral Centroid</div>
                        <div className="feature-value">{pred.spectral_centroid?.toFixed(2)} Hz</div>
                      </div>
                      <div className="feature-card">
                        <div className="feature-label">Zero Crossing Rate</div>
                        <div className="feature-value">{pred.zero_crossing_rate?.toFixed(4)}</div>
                      </div>
                      <div className="feature-card">
                        <div className="feature-label">Energy</div>
                        <div className="feature-value">{pred.energy?.toFixed(4)}</div>
                      </div>
                      <div className="feature-card">
                        <div className="feature-label">MFCC Features</div>
                        <div className="feature-value">13 coefficients</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;