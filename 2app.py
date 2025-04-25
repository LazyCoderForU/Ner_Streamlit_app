import streamlit as st
import pickle
import nltk
import spacy
from sklearn_crfsuite import CRF

# Download resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load CRF model
with open('crf_model.pkl', 'rb') as f:
    crf = pickle.load(f)

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Feature extraction
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'word.isalpha()': word.isalpha(),
        'word.isalnum()': word.isalnum(),
        'word.startswith.upper()': word[0].isupper(),
        'word.endswith.s': word.endswith('s'),
        'word.length': len(word),
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:word.isalpha()': word1.isalpha(),
            '-1:word.isalnum()': word1.isalnum(),
            '-1:word.startswith.upper()': word1[0].isupper(),
            '-1:word.endswith.s': word1.endswith('s'),
            '-1:word.length': len(word1),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:word.isalpha()': word1.isalpha(),
            '+1:word.isalnum()': word1.isalnum(),
            '+1:word.startswith.upper()': word1[0].isupper(),
            '+1:word.endswith.s': word1.endswith('s'),
            '+1:word.length': len(word1),
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

# Color map for NER tags
color_map = {
    'geo': '#F94144',
    'org': '#F3722C',
    'per': '#F8961E',
    'gpe': '#F9844A',
    'tim': '#90BE6D',
    'art': '#43AA8B',
    'eve': '#577590',
    'nat': '#277DA1',
    'O': '#E0E0E0'  # grey for non-entities
}

# Streamlit UI
st.title("📝 CRF Named Entity Recognition (NER) Web App")

sentence = st.text_area("Enter a sentence for NER", 
                        "India is going to win the Apple stocks and can get a profit of 2 billion dollars in the next year 2020 with 2kg of apples")

if st.button("Analyze NER"):

    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)

    X_input = sent2features(pos_tags)
    ner_tags = crf.predict([X_input])[0]

    st.subheader("🔍 NER Tags:")

    st.markdown("""
    <style>
    .ner-card {
        display: inline-block;
        padding: 6px 10px;
        margin: 4px 4px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Display words as colored cards with tags
    output_html = ""
    for word, tag in zip(tokens, ner_tags):
        bg_color = color_map.get(tag, "#888888")
        output_html += f'<span class="ner-card" style="background-color:{bg_color}">{word} <small>({tag})</small></span> '

    st.markdown(output_html, unsafe_allow_html=True)

    st.subheader("📊 Token-wise Predictions")
    for word, tag in zip(tokens, ner_tags):
        st.write(f"**{word}** → _{tag}_")

