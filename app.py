import streamlit as st
import pickle
import nltk
import spacy
from sklearn_crfsuite import CRF
from spacy import displacy

# Download necessary resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load CRF model
with open('crf_model.pkl', 'rb') as f:
    crf = pickle.load(f)

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Feature extraction functions
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
st.title("📝 CRF-based Named Entity Recognition")

sentence = st.text_area("Enter a sentence for NER", 
                        "India is going to win the Apple stocks and can get a profit of 2 billion dollars in the next year 2020 with 2kg of apples")

if st.button("Analyze NER"):

    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)

    X_input = sent2features(pos_tags)
    ner_tags = crf.predict([X_input])[0]

    st.subheader("🔍 Token-wise NER result:")
    for token, tag in zip(tokens, ner_tags):
        st.write(f"**{token}** → _{tag}_")

    st.subheader("🎨 Visualized NER using spaCy:")
    # Create a spaCy Doc object manually
    doc = nlp(sentence)
    ents = []
    start = 0

    for token, label in zip(doc, ner_tags):
        end = start + len(token.text)
        if label != 'O':
            ents.append({'start': start, 'end': end, 'label': label})
        start = end + 1  # account for space

    doc.ents = [spacy.tokens.Span(doc, doc.char_span(ent['start'], ent['end']).start, doc.char_span(ent['start'], ent['end']).end, label=ent['label']) for ent in ents if doc.char_span(ent['start'], ent['end'])]

    html = displacy.render(doc, style="ent", options={"colors": {
        "geo": "#e76f51",
        "org": "#2a9d8f",
        "per": "#f4a261",
        "gpe": "#e9c46a",
        "tim": "#264653",
        "art": "#a8dadc",
        "eve": "#457b9d",
        "nat": "#ffafcc"
    }}, page=True)
    
    st.components.v1.html(html, height=300, scrolling=True)

