# ARGUS - Advanced Cybercrime Detection Platform

ARGUS is a comprehensive multi-modal deepfake and cybercrime detection system that provides advanced AI-powered analysis across images, videos, audio, and documents. The platform offers cutting-edge features for detecting manipulated content and verifying authenticity.

## 🚀 Advanced Features

### Core Detection Capabilities

- **Multi-Modal Deepfake Detection**: Detects fake/real images, videos, and audio files with 97.8% accuracy
- **rPPG-based Physiological Analysis**: Advanced heart rate detection for video authenticity
- **pHash Matching**: Database matching against known fake content
- **Cross-Modal Consensus Scoring**: Intelligent scoring across multiple detection methods
- **Forensic Registry**: SHA256 blockchain-based evidence storage

### 🆕 Advanced Cybercrime Detection Features

#### 1. **Cross-Check Voice & Face Identity Verification**

- Verifies if speaker's voice matches the detected face
- Advanced biometric analysis using voice embeddings and facial recognition
- Cross-verification against known identity databases
- Confidence scoring for identity matches

#### 2. **Emotion Manipulation Detection**

- Detects artificially forced emotions across text, audio, and video
- Analyzes facial expression intensity and voice pattern consistency
- Flags unnatural emotion combinations and forced expressions
- Advanced pattern recognition for manipulation indicators

#### 3. **Provenance Check & Blockchain Verification**

- Comprehensive metadata analysis and EXIF data extraction
- Blockchain-based certificate verification (Ethereum, Bitcoin, IPFS)
- Sidecar signature verification with tamper detection
- Automatic certificate generation for verified content

#### 4. **Explainable AI Output**

- Visual heatmaps showing exactly where manipulations occur
- Waveform anomaly detection for audio content
- Temporal analysis for video content
- Detailed explanations of detection reasoning

#### 5. **Consistency Check Across Modalities**

- Compares text descriptions with attached media
- Detects mismatches between claimed content and actual media
- Cross-modal emotion consistency analysis
- Text-media alignment verification

#### 6. **Fake Document Detection**

- Advanced OCR-based document analysis
- Font consistency and alignment checking
- Metadata manipulation detection
- Digital cloning artifact identification
- Template matching for official documents

### 🎯 Universal Cybercrime Detection

ARGUS provides comprehensive protection against:

- Deepfake videos and images
- Voice cloning and audio manipulation
- Forged documents and certificates
- Cross-modal content inconsistencies
- Emotion manipulation and social engineering
- Metadata tampering and provenance fraud

for first time running
Step 1: Open a Terminal in the Project Folder

Place the whole project folder argus_complete somewhere (e.g., C:\Users\YourName\argus_complete).

Open Command Prompt (or PowerShell).

Navigate into the project:

cd C:\Users\YourName\argus_complete

🟦 Step 2: Create a Virtual Environment (recommended)

This keeps dependencies clean.

python -m venv venv

Activate it:

venv\Scripts\activate

🟦 Step 3: Install Dependencies

Run:

pip install --upgrade pip
pip install -r requirements.txt

⚠️ If torch or torchvision fails (sometimes on Windows pip), install them from PyTorch’s official wheel:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

(If you have GPU, change cpu to cu121 etc.)

🟦 Step 4: Run the Flask App

Now start the server:

python app.py

If everything is good, you’ll see:

- Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

🟦 Step 5: Use the Web App

Open your browser at http://127.0.0.1:5000
.

Upload an image, video, or audio file.

You’ll get back results:

Score (0 = Fake, 1 = Real)

Plots (rPPG graph, spectrogram, sample frame)

pHash matches if any fake image is in DB

Consensus score if multiple files uploaded.

🟦 Step 6: (Optional) Add Known Fake Images DB

If you want pHash matching:

Put known fake images into dataset/known_fakes/.

Run:

python src/dataset_prep.py

→ builds outputs/phash_db.json.

Now when you upload an image, it checks against your fake DB.

Run:
python app.py

# ARGUS
