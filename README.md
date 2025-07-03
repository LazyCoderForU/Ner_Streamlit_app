# 📝 Named Entity Recognition (NER) using CRF & Streamlit

Welcome to the **NER App** — a powerful, interactive tool to detect and visualize named entities (like 👤 persons, 🌍 locations, 🏢 organizations, and 🕒 time references) in text! Built using a **Conditional Random Field (CRF)** model, this web app leverages **NLTK**, **spaCy**, and **Streamlit** to provide an intuitive experience.

---

## 📂 Dataset

The model is trained on a dataset containing over **1 million rows** in the following format:

```
Sentence #     Word     POS     Tag
Sentence: 1    London   NNP     B-geo
              is       VBZ     O
              ...      ...     ...
```

Each word is labeled with:
- **POS tag**: Part-of-Speech (like noun, verb, etc.)
- **NER tag**: Named Entity type (like B-geo for location)

---

## 🛠️ Features

✅ CRF-based NER tagger  
✅ Preprocessing with **NLTK** and **spaCy**  
✅ Live entity prediction using a **Streamlit** app  
✅ Friendly labels and color-coded visualization  
✅ Table view for detailed output  

---

## 🎨 Tag Labels & Emojis

| Tag | Meaning | Emoji |
|-----|---------|-------|
| B-geo / I-geo | Geographical Location | 🌍 |
| B-org / I-org | Organization | 🏢 |
| B-per / I-per | Person | 👤 |
| B-gpe / I-gpe | Geopolitical Entity | 🗺️ |
| B-tim / I-tim | Time Expression | 🕒 |
| B-art / I-art | Artifact | 🎨 |
| B-eve / I-eve | Event | 🎉 |
| B-nat / I-nat | Natural Phenomenon | 🌋 |
| O | Other | ⚪ |

---

## 🚀 How to Run the App

### 🔧 Setup

1. Clone the repo and navigate to the folder:
   ```bash
   git clone https://github.com/your-username/ner-crf-app.git
   cd ner-crf-app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download and link the **spaCy** model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. Launch the app:
   ```bash
   streamlit run 63332ae4-9a6a-45f2-8609-7284f8db335e.py
   ```

---

## 🧠 Model

- Model: `CRF` (Conditional Random Field)
- Features extracted:
  - Word shape, prefix/suffix, POS tags, surrounding context
- Tagging follows BIO (Beginning-Inside-Outside) scheme

---

## 🗃️ Files Included

- `ner_dataset.csv` → Large labeled dataset
- `crf_model.pkl` → Pre-trained CRF model
- `63332ae4-...py` → Streamlit frontend app
- `requirements.txt` → List of dependencies
- `README.md` → You're reading it! 😄

---

## 🛠️ Tech Stack

- **Programming Language**: Python 🐍
- **Libraries**: NLTK, spaCy, Streamlit, sklearn-crfsuite
- **Frontend**: Streamlit for interactive UI
- **Backend**: CRF model for NER tagging
- **Visualization**: Color-coded entity highlights

---

## 🔄 Workflow / How This Project Was Made

1. **Data Preparation**:
   - Collected and preprocessed a large dataset with over 1 million rows.
   - Labeled data using BIO (Beginning-Inside-Outside) scheme for NER.

2. **Feature Engineering**:
   - Extracted features like word shape, prefix/suffix, POS tags, and surrounding context.

3. **Model Training**:
   - Trained a Conditional Random Field (CRF) model using sklearn-crfsuite.
   - Tuned hyperparameters for optimal performance.

4. **Frontend Development**:
   - Built an interactive web app using Streamlit.
   - Integrated the CRF model for live predictions.

5. **Visualization**:
   - Added color-coded highlights for entity types.
   - Designed a table view for detailed output.

6. **Deployment**:
   - Packaged the app with all dependencies.
   - Provided instructions for easy setup and usage.

---

## 👨‍💻 Author

Made with ❤️ by **Brajesh Kumar**  
📧 brajeshguptaa1@gmail.com  


---

## 🏁 Future Improvements

- 📈 Add model training notebook  
- 🧪 Support evaluation metrics (precision, recall, F1)  
- 🖼️ Highlight entities directly on uploaded documents
