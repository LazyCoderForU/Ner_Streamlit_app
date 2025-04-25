import streamlit as st
import pickle
import nltk
import spacy
from sklearn_crfsuite import CRF
from spacy import displacy

# Load CRF model
with open('crf_model.pkl', 'rb') as f:
    crf = pickle.load(f)

# Load spaCy model for POS tagging
nlp = spacy.load("en_core_web_sm")

# Function to convert spaCy pos_ to readable names
def pos_readable(pos):
    pos_map = {
        'NOUN': 'Noun',
        'VERB': 'Verb',
        'PROPN': 'Proper Noun',
        'ADJ': 'Adjective',
        'ADV': 'Adverb',
        'DET': 'Determiner',
        'ADP': 'Preposition',
        'NUM': 'Number',
        'PRON': 'Pronoun',
        'CCONJ': 'Conjunction',
        'PART': 'Particle',
        'PUNCT': 'Punctuation',
        'SYM': 'Symbol',
        'X': 'Other',
        'SPACE': 'Space'
    }
    return pos_map.get(pos, pos)

# Mapping of NER tags to emoji labels
tag_legend = {
    'O': '⚪ Other',
    'B-geo': '🌍 Geographical Location',
    'B-org': '🏢 Organization',
    'B-per': '👤 Person',
    'B-gpe': '🌏 Geopolitical Entity',
    'B-tim': '🕒 Time',
    'B-art': '🎨 Artifact',
    'B-eve': '🎉 Event',
    'B-nat': '🌋 Natural Phenomenon'
}

# NER Features extraction
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
        'postag[:2]': postag[:2]
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2]
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
            '+1:postag[:2]': postag1[:2]
        })
    else:
        features['EOS'] = True
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

# Streamlit UI
st.title("📝 Named Entity Recognition (NER) App")
st.write("Type any sentence below and this app will detect names, places, organizations, dates — and show grammar details too!")

sentence = st.text_input("✍️ Enter a sentence:", "Barack Obama visited Paris in July 2021.")

if sentence:
    doc = nlp(sentence)
    pos_tags = [(token.text, token.pos_) for token in doc]
    pos_tagged = [(word, pos) for word, pos in pos_tags]

    # Prepare CRF input
    crf_input = [(word, pos) for word, pos in pos_tags]
    features = sent2features(crf_input)
    ner_tags = crf.predict([features])[0]

    # Display results
    st.markdown("### 📊 NER Results:")
    for (word, pos), tag in zip(pos_tags, ner_tags):
        ner_label = tag_legend.get(tag, '🔘 Unknown')
        pos_label = pos_readable(pos)
        st.markdown(f"**{word}** → {ner_label} <span style='color:gray'>({pos_label})</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Detailed Table View:")
    for (word, pos), tag in zip(pos_tags, ner_tags):
        ner_label = tag_legend.get(tag, '🔘 Unknown')
        pos_label = pos_readable(pos)
        st.markdown(f"{word} → {ner_label} <span style='color:gray'>({pos_label})</span>", unsafe_allow_html=True)

    st.markdown("Made with ❤️ using CRF and Streamlit")

