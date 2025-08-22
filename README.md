# Resource Allocation using Multimodal AI with Misinformation Mitigation in Healthcare
This project presents a comprehensive multimodal AI framework designed to predict mental health severity, particularly in the context of Alzheimer's disease, by leveraging diverse data sources:

* Neuroimaging: Structural and functional data from MRI and PET scans
* Vocal Biomarkers: Audio features (e.g., MFCCs) capturing speech characteristics correlated with cognitive decline
* Physiological & Behavioural Signals: Including gaze patterns, body pose, breathing rate, heart rate, and motor tapping
* Clinical Scores: Such as PHQ scores, indicating depression and mood-related symptoms

These multimodal inputs are used to train a predictive model that estimates cognitive severity, supporting early intervention and optimised care.
Beyond the clinical prediction model, this project also simulates the spread of health misinformation via social media networks, modelling how exposure to inaccurate or harmful information can impact perception and decision-making. This simulation is used to adjust severity predictions and dynamically inform resource allocation — ensuring patients most at risk (both medically and informationally) are prioritised for treatment.

The entire pipeline is visualised through a Streamlit app, including model predictions, explainability tools (SHAP, LIME), and a real-time network-based misinformation simulation.

## Key Features
* Multimodal Input: Combines features from audio (MFCC), image (ResNet), physiological signals, gaze, pose, and PHQ scores.
* Severity Prediction: Uses Random Forest Regression to predict PHQ severity.
* Misinformation Spread Simulation: Network-based misinformation modelling to adjust severity scores.
* Resource Allocation: Dynamically allocate limited treatment resources based on adjusted scores.
* Explainability: Visual explanations using SHAP and LIME for transparent predictions.
* Web App: Interactive UI built with Streamlit for simulations, explanations, and visualisations.

## Project Structure
```plaintext
project/
│
├── audio_features.npy
├── image_features.npy
├── physio_features.npy
├── gaze_features.npy
├── pose_features.npy
├── phq_scores.npy
├── train_phq_score_*.npy
│
├── models/
│   └── severity_model.pkl
│
├── validation_plot.png
└── script_name.py 
```

## Installation
### 1. Clone the Repository
```plaintext
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```
### 2. Install Dependencies
```plaintext
Use pip to install all required packages:
pip install streamlit librosa torch torchvision shap lime scikit-learn matplotlib networkx pandas pillow
```
## Running the App
Once dependencies are installed and your .npy data files are in place:
```plaintext
streamlit run your_script_name.py
```
Then open the local Streamlit URL (usually http://localhost:8501) in your browser.

## How It Works
### 1. Feature Extraction
extract_mfcc_features(audio_files) – Audio MFCC extraction via librosa
extract_resnet_features(image_files) – Image embeddings using pretrained ResNet-18
Physiological markers are synthetically simulated (breathing, tapping, heart rate)

### 2. Model Training
* All features are concatenated and standardised
* Random Forest Regressor trained to predict PHQ scores
* Model performance validated using MAE, RMSE and R²

### 3. Misinformation Simulation
Simulates misinformation spread using a Barabási–Albert network model:

* Nodes: Patients
* States: Susceptible (S), Infected (I) and Recovered (R)
* Adjusts predicted severity scores based on misinformation prevalence

### 4. Resource Allocation
* Patients with the highest adjusted severity are prioritised
* Limited capacity is configurable via the Streamlit app

### 5. Explainability
* SHAP: Visualises individual feature contributions for predictions
* LIME: Explains local model behaviour for a selected patient

## Validation
* Validation plot (validation_plot.png) shows predicted vs actual PHQ severity on the test set.
* Automatically displayed in the Streamlit app.

## Example Outputs
* PHQ Score Prediction Graphs
* SHAP & LIME Patient Explanation Visuals
* Misinformation Network Evolution
* Risk-Based Resource Allocation Charts

## Notes
* Ensure all .npy feature files are correctly shaped and aligned.
* The train_phq_score_*.npy files are merged to generate ground truth severity labels.
* Misinformation risk is integrated directly into adjusted predictions.

## Acknowledgments
* librosa – Audio analysis
* torchvision – Pretrained image models
* SHAP\LIME – Model explainability
* Streamlit – Interactive app
* scikit-learn – ML modelling

## Contact
For questions or collaboration requests, please contact me here or open an issue.

