import streamlit as st
import pickle
import nltk
import en_core_web_sm
from nltk import pos_tag, word_tokenize

# Download NLTK resources if missing
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load CRF model
with open('crf_model.pkl', 'rb') as f:
    crf = pickle.load(f)

# Load spaCy model (pre-installed via requirements.txt)
nlp = en_core_web_sm.load()

# Friendly labels for NER tags
friendly_labels = {
    'geo': '🌍 Geographical Location',
    'org': '🏢 Organization',
    'per': '👤 Person',
    'gpe': '🗺️ Geopolitical Entity',
    'tim': '🕒 Time',
    'art': '🎨 Artifact',
    'eve': '🎉 Event',
    'nat': '🌋 Natural Phenomenon',
    'O': '⚪ Other'
}

# Color map for nice visuals
color_map = {
    'geo': '#8BC34A',
    'org': '#FF9800',
    'per': '#03A9F4',
    'gpe': '#9C27B0',
    'tim': '#FFC107',
    'art': '#E91E63',
    'eve': '#673AB7',
    'nat': '#009688',
    'O': '#B0BEC5'
}

# Feature extractor functions
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

# Streamlit UI
st.title("📝 Named Entity Recognition (NER) App")
st.write("Type any sentence below and this app will detect names, places, organizations, dates, and more — with easy-to-understand labels!")

# User input
sentence = st.text_input("✍️ Enter a sentence:", "India is going to win the Apple stocks in 2020")

if st.button("🔍 Analyze"):
    # Tokenize and POS tag the sentence
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)

    # Predict NER tags
    ner_tags = crf.predict([sent2features(pos_tags)])[0]

    # Display the result nicely
    st.markdown("### 📊 NER Results:")

    output_html = ""
    for word, tag in zip(tokens, ner_tags):
        clean_tag = tag.replace("B-", "").replace("I-", "")
        label = friendly_labels.get(clean_tag, "🔘 Unknown")
        bg_color = color_map.get(clean_tag, "#888888")
        output_html += f'<span style="background-color:{bg_color}; color:white; padding:4px 8px; margin:4px; border-radius:8px; display:inline-block;">{word}<small style="display:block; font-size:10px;">{label}</small></span> '

    st.markdown(output_html, unsafe_allow_html=True)

    # Show the tokens + tags as a table too
    st.markdown("### 📋 Detailed Table View:")
    for word, tag in zip(tokens, ner_tags):
        clean_tag = tag.replace("B-", "").replace("I-", "")
        label = friendly_labels.get(clean_tag, "🔘 Unknown")
        st.write(f"**{word}** → {label}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using CRF and Streamlit")
