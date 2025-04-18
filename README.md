
# Named Entity Recognition (NER) with CRF Model

This project implements a Named Entity Recognition (NER) system using a Conditional Random Field (CRF) model to identify named entities such as geographical locations, organizations, persons, dates, and more within text. The system is trained on a labeled dataset containing various named entity types (e.g., **geo**, **org**, **per**, **gpe**, **tim**, etc.) and can predict these entities in new, unseen sentences.



## Prerequisites

Ensure you have the following Python libraries installed:

pip install sklearn-crfsuite scikit-learn pandas nltk spacy
Additionally, you will need to download the en_core_web_sm model for spaCy:

python -m spacy download en_core_web_sm
Dataset
The dataset ner_dataset.csv contains sentences labeled with various entity types (e.g., geo, org, per, gpe, tim, etc.). It is structured as follows:


Sentence #	Word	POS	Tag
Sentence: 1	Thousands	NNS	O
...	...	...	...
Sentence #: Unique identifier for each sentence.

Word: A single word in the sentence.

POS: Part of Speech tag.

Tag: NER label (e.g., B-geo, B-org, O, etc.).

Data Preprocessing
The data preprocessing steps include:

Handling missing values: The 'Sentence #' column is filled using forward filling (ffill method).

Grouping by sentence: The words are grouped by sentence, with each sentence represented as a list of tuples containing the word, its part of speech (POS) tag, and the corresponding NER label.

Feature Engineering: Various features (e.g., word case, POS, neighboring words) are extracted from each word in the sentence for model training.

Model Training
The CRF model is trained using the following steps:

Feature Extraction: Features such as word characteristics, POS tags, and neighboring words are extracted for each word in the sentence.

Model Training: A Conditional Random Field (CRF) model is trained using the sklearn-crfsuite library. The following hyperparameters are used:

Algorithm: l2sgd (Stochastic Gradient Descent with L2 regularization)

Regularization: c2 = 0.1

Max Iterations: 100

Saving the Model: The trained model is saved as crf_model.pkl using the pickle library for future use.

Model Evaluation
The model is evaluated using a test set, with the following metrics computed:

F1-score (weighted)

Precision

Recall

Accuracy

Sample code for evaluating the model:

python
Copy
Edit
f1_score = flat_f1_score(y_test, y_pred, average='weighted')
print(f1_score)

flat_f1_score(y_test, y_pred, average='weighted')
flat_precision_score(y_test, y_pred, average='weighted')
sequence_accuracy_score(y_test, y_pred)
flat_recall_score(y_test, y_pred, average='weighted')
flat_accuracy_score(y_test, y_pred)
report = flat_classification_report(y_test, y_pred)
print(report)
Using the Trained CRF Model
Once the model is trained, it can be used to predict named entities in new sentences. The steps to predict are:

Tokenize the input sentence using nltk.

POS tag the tokens using nltk.

Extract features from the POS-tagged tokens.

Predict the entities using the trained CRF model.

Example of predicting named entities:

python
Copy
Edit
sentence = "India is going to win the Apple stocks and can get a profit of 2 billion dollars in the next year 2020 with 2kg of apples"
tokens = nltk.word_tokenize(sentence)
pos_tags = nltk.pos_tag(tokens)

# Predict named entities using the trained CRF model
ner_tags = crf.predict([sent2features(pos_tags)])[0]

# Visualize the results using spaCy
import spacy
from spacy import displacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Create a spaCy doc from the sentence
doc = nlp(sentence)
for token, ner_tag in zip(doc, ner_tags):
    token.ent_type_ = ner_tag

# Render the named entities in the sentence
displacy.render(doc, style="ent", jupyter=True)
Visualization
The named entities in the sentence can be visualized using the spaCy displacy tool. The entities will be highlighted in the rendered sentence.

python
Copy
Edit
displacy.render(doc, style='dep', jupyter=True, options={
    "compact": True, "bg": "#09a3d5", "color": "white", "font": "Source Sans Pro", "fine_grained": True
})
Conclusion
This project demonstrates how to build and evaluate a CRF-based Named Entity Recognition system. The trained model is capable of recognizing and classifying named entities in text. The model can be extended further by adding more advanced features, fine-tuning hyperparameters, or experimenting with other machine learning models.

Future Work
Hyperparameter Tuning: Optimize the CRF model by tuning its hyperparameters for better performance.

Advanced Features: Integrate more sophisticated features like word embeddings (e.g., GloVe, Word2Vec).

Deep Learning Models: Explore the use of deep learning models like BERT for NER.

Fine-tuning Pretrained Models: Fine-tune pre-trained NER models (like spaCy's NER model) for the custom dataset.

License
This project is licensed under the MIT License.

This `README.md` file includes an explanation of the project, setup instructions, how to use the model, and evaluation methods. It also provides a clear structure and an outline for future work. You can place this content in the `README.md` file of your repository.







