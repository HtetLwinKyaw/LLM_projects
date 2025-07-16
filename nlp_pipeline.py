# Required installations:
# pip install spacy nltk
# python -m spacy download en_core_web_sm

import spacy
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import treebank
from nltk.tree import Tree
from spacy import displacy

# Download required NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('treebank')

# Sample text for both tools
sample_text = "Apple is looking at buying U.K. startup for $1 billion."

print("="*80)
print("📌 Using spaCy")
print("="*80)

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Run the NLP pipeline
doc = nlp(sample_text)

# Tokenization
print("\n🔹 Tokenization:")
print([token.text for token in doc])

# Part-of-Speech Tagging
print("\n🔹 POS Tagging:")
print([(token.text, token.pos_) for token in doc])

# Named Entity Recognition (NER)
print("\n🔹 Named Entities:")
for ent in doc.ents:
    print((ent.text, ent.label_))

# Lemmatization
print("\n🔹 Lemmatization:")
print([(token.text, token.lemma_) for token in doc])

# Dependency Parsing
print("\n🔹 Dependency Parsing:")
print([(token.text, token.dep_, token.head.text) for token in doc])

# Visualize NER using displacy (for notebook / HTML view)
print("\n🔹 Visualizing Named Entities (Open in browser or render in notebook):")
displacy.serve(doc, style="ent")


print("\n\n" + "="*80)
print("📌 Using NLTK")
print("="*80)

# Tokenization
tokens = word_tokenize(sample_text)
print("\n🔹 Tokenization:")
print(tokens)

# POS Tagging
pos_tags = pos_tag(tokens)
print("\n🔹 POS Tagging:")
print(pos_tags)

# Named Entity Recognition (using ne_chunk)
print("\n🔹 Named Entities:")
ne_tree = ne_chunk(pos_tags)
for subtree in ne_tree:
    if isinstance(subtree, Tree):
        entity_name = " ".join(c[0] for c in subtree)
        entity_type = subtree.label()
        print((entity_name, entity_type))

# Extra: Compare outputs
print("\n\n" + "="*80)
print("📊 Comparison of spaCy and NLTK on Tokenization + POS + NER")
print("="*80)

print("\n[spaCy Tokens]")
print([token.text for token in doc])
print("\n[NLTK Tokens]")
print(tokens)

print("\n[spaCy POS Tags]")
print([(token.text, token.pos_) for token in doc])
print("\n[NLTK POS Tags]")
print(pos_tags)

print("\n[spaCy Named Entities]")
print([(ent.text, ent.label_) for ent in doc])
print("\n[NLTK Named Entities]")
for subtree in ne_tree:
    if isinstance(subtree, Tree):
        entity_name = " ".join(c[0] for c in subtree)
        entity_type = subtree.label()
        print((entity_name, entity_type))
