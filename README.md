# EEG-Seizure-Detector
🧠 NeuroScan · EEG Seizure Detector
NeuroScan is a high-fidelity Streamlit dashboard designed to bridge the gap between complex machine learning and accessible health insights. It analyzes EEG (Electroencephalogram) brain signals using three distinct AI models to identify potential seizure activity.

✨ Key Features
Real-time ML Analysis: Choose between Random Forest, SVM, and K-NN models.

Plain English Insights: Deep technical metrics (F1, MCC, Kappa) explained in simple terms.

Interactive Simulation: Adjust 12 different brain signal features (Delta, Theta, Alpha, Beta, Gamma, etc.) to see how the AI responds.

Comparative Performance: Side-by-side model rankings and cross-validation results.

Modern UI: A dark-themed, glassmorphic interface built with custom CSS and the "DM Sans" font.

🔬 Signal Features Tracked
The AI evaluates the following brain wave bands and signal characteristics:

Frequency Bands: Delta (Deep sleep/Seizure drop), Alpha (Relaxed), Gamma (Seizure spike).

Signal Geometry: Line Length and Variance.

Complexity: Sample Entropy and Kurtosis.

🚀 Getting Started
1. Prerequisites
Ensure you have Python 3.8+ installed. You will need the following libraries:

Bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
2. Installation
Clone this repository or download the source code.

Navigate to the project folder:

Bash
cd neuroscan-healthtech
3. Run the Dashboard
Launch the application using Streamlit:

Bash
streamlit run app.py
📊 AI Methodology
Training Set: Synthetic EEG data modeled after 2,500 clinical windows (2,000 Normal / 500 Seizure).

Validation: 5-Fold Stratified Cross-Validation to ensure model stability.

Standardization: All inputs are passed through a StandardScaler to ensure the models interpret voltage and frequency fluctuations accurately.

⚠️ Medical Disclaimer
NeuroScan is a research and awareness tool. It is not a certified medical device and is not intended to provide a formal medical diagnosis. All findings should be reviewed by a qualified neurologist.
<img width="1910" height="922" alt="image" src="https://github.com/user-attachments/assets/cb844575-7469-4d90-9b6d-2afda1ee23f1" />
<img width="1882" height="912" alt="image" src="https://github.com/user-attachments/assets/4dafa0b7-80a0-47e1-a149-3f96261f21f5" />

