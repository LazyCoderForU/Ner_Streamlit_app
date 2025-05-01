# 📝 Named Entity Recognition (NER) using CRF & Streamlit

Welcome to the **NER App** — a powerful, interactive tool to detect and visualize named entities (like 👤 persons, 🌍 locations, 🏢 organizations, and 🕒 time references) in text! Built using a **Conditional Random Field (CRF)** model, this web app leverages **NLTK**, **spaCy**, and **Streamlit** to provide an intuitive experience.

---

## 📝 Description

The **NER App** is designed to:
- Detect named entities in text using a CRF model.
- Provide a user-friendly interface for entity visualization.
- Solve the problem of manual entity recognition by automating the process with machine learning.

Key Features:
- CRF-based NER tagger.
- Preprocessing with **NLTK** and **spaCy**.
- Live entity prediction using a **Streamlit** app.
- Color-coded visualization for better understanding.
- Table view for detailed output.

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

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ner-streamlit-app.git
   cd ner-streamlit-app
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download and link the **spaCy** model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

---

## 📦 Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open the provided local URL in your browser to interact with the app.

---

## 🖼️ Screenshots / Demo

![NER App Demo](https://via.placeholder.com/800x400.png?text=Demo+Screenshot)

---

## 🧠 Tech Stack / Built With

- **Python**
- **Streamlit**
- **spaCy**
- **NLTK**
- **CRF Suite**

---

## 📂 Project Structure

```
.
├── app.py                # Main Streamlit app
├── main_app.py           # Additional app logic
├── CRF model for Ner.ipynb  # Jupyter Notebook for model training
├── crf_model.pkl         # Pre-trained CRF model
├── ner_dataset.csv       # Dataset for training
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── myenv/                # Virtual environment
```

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

## 🙌 Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch-name`).
5. Open a pull request.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

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

## 👨‍💻 Author

Made with ❤️ by **Brajesh Kumar**  
📧 brajesh350194@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/brajesh-gupta) | 💻 [GitHub](https://github.com/LazyCoderForU)

---

## 🏁 Future Improvements

- 📈 Add model training notebook  
- 🧪 Support evaluation metrics (precision, recall, F1)  
- 🖼️ Highlight entities directly on uploaded documents
