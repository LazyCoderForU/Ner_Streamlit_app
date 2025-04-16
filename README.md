# Named Entity Recognition (NER) with CRF and Streamlit

This repository demonstrates the development of a Named Entity Recognition (NER) application using Conditional Random Fields (CRF) and Streamlit. The project includes multiple variations of the app, each with unique features and enhancements.

## Overview of Files

### Main Files
- **main_app.py**: The primary application showcasing NER using CRF and Streamlit.
- **2app.py, 3app.py, 4app.py, 5app.py**: Variations of the main app with additional features and visualizations.

### Supporting Files
- **crf_model.pkl**: Pre-trained CRF model used for NER.
- **ner_dataset.csv**: Dataset used for training the CRF model.
- **CRF model for Ner.ipynb**: Jupyter Notebook detailing the training process of the CRF model.

## How the Apps Were Built

### main_app.py
1. **Libraries Used**:
   - `streamlit` for building the web interface.
   - `nltk` and `spacy` for tokenization and POS tagging.
   - `pickle` for loading the pre-trained CRF model.
2. **Features**:
   - Accepts user input for sentences.
   - Uses the CRF model to predict NER tags.
   - Displays results with user-friendly labels and color-coded tags.
3. **Visualization**:
   - Inline HTML rendering for NER results.
   - Detailed table view of tokens and their corresponding tags.

### Variations (2app.py, 3app.py, etc.)
Each variation builds upon the main app with additional features:

- **2app.py**:
  - Simplified UI with basic NER visualization.
  - Focuses on token-wise predictions.

- **3app.py**:
  - Enhanced color-coded visualization for NER tags.
  - Includes a legend for tag meanings.

- **4app.py**:
  - Adds POS tagging details alongside NER results.
  - Displays grammar details for each token.

- **5app.py**:
  - Combines NER and POS tagging with advanced visualizations.
  - Uses emoji-based labels for better user experience.

## Training the CRF Model
The CRF model was trained using the `ner_dataset.csv` dataset. The training process is documented in the Jupyter Notebook `CRF model for Ner.ipynb`. Key steps include:

1. Data preprocessing and feature extraction.
2. Splitting the dataset into training and testing sets.
3. Training the CRF model using `sklearn-crfsuite`.
4. Evaluating the model's performance using metrics like F1-score and accuracy.

## How to Run the Apps
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run any of the apps using Streamlit:
   ```bash
   streamlit run main_app.py
   ```
   Replace `main_app.py` with the desired variation (e.g., `2app.py`).

## Future Enhancements
- Add support for additional NER tags.
- Improve the UI/UX with more interactive elements.
- Integrate with external APIs for real-time data processing.
