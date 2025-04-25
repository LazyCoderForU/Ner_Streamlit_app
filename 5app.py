import streamlit as st
import pickle
import spacy
from sklearn_crfsuite import CRF

# Load CRF model
with open('crf_model.pkl', 'rb') as f:
    crf = pickle.load(f)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# NER tag to label + color
tag_legend = {
    'O': {'label': '', 'color': '#888888'},  # Grey for non-entity POS
    'B-geo': {'label': '🌍 Location', 'color': '#06D6A0'},
    'B-org': {'label': '🏢 Organization', 'color': '#118AB2'},
    'B-per': {'label': '👤 Person', 'color': '#FFD166'},
    'B-gpe': {'label': '🌏 Country', 'color': '#06D6A0'},
    'B-tim': {'label': '🕒 Time', 'color': '#EF476F'},
    'B-art': {'label': '🎨 Artifact', 'color': '#8E44AD'},
    'B-eve': {'label': '🎉 Event', 'color': '#FFA07A'},
    'B-nat': {'label': '🌋 Nature', 'color': '#A9A9A9'}
}

# POS readable mapping
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

# CRF features
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
st.title("📝 NER + POS Visualizer")
st.write("Type any sentence below — it'll highlight names, places, times with color, and show other words by their part of speech.")

sentence = st.text_input("✍️ Enter a sentence:", "Barack Obama visited Paris in July 2021.")

if sentence.strip() != "":
    doc = nlp(sentence)
    pos_tags = [(token.text, token.pos_) for token in doc]
    crf_input = [(word, pos) for word, pos in pos_tags]
    features = sent2features(crf_input)
    ner_tags = crf.predict([features])[0]

    # Visual Inline Text Rendering
    st.markdown("### 📊 Visual Result:")

    result_html = ""
    for (word, pos), tag in zip(pos_tags, ner_tags):
        tag_info = tag_legend.get(tag, {'label': '🔘 Unknown', 'color': '#888888'})
        color = tag_info['color']
        label = tag_info['label']

        if tag == 'O':
            label = pos_readable(pos)
        result_html += f"<span style='background-color:{color}; color:white; padding:2px 6px; border-radius:4px; margin:2px'>{word} <small>({label})</small></span> "

    st.markdown(result_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Full Table:")

    for (word, pos), tag in zip(pos_tags, ner_tags):
        tag_info = tag_legend.get(tag, {'label': '🔘 Unknown', 'color': '#888888'})
        ner_label = tag_info['label'] if tag != 'O' else 'Other'
        pos_label = pos_readable(pos)
        st.markdown(f"**{word}** → {ner_label} <span style='color:gray'>({pos_label})</span>", unsafe_allow_html=True)

    st.markdown("Made with ❤️ using CRF, spaCy and Streamlit")

