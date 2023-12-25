import spacy
from spacy.cli import download

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # download en_core_web_sm
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

texts = [
    "what have I written about sonification?",
    "what do my notes say about sonification?",
    "tell me about sonification",
    "what is sonification?",
    "for my IMC work, what are my hours?",
    "summarise my IMC work",
    "summarise my work for IMC",
    "summarise writings about sonification",
    "summarise Natural Language Processing notes",
    "summarise natural language processing",
    "summarise advanced cognitive neuroscience",
    "summarise philosophy of cognitive science",
    "summarise my notes on philosophy of cognitive science",
]

what_pos = ["NOUN", "PROPN", "ADP"]
where_pos = ["VERB", "AUX", "ADP"]


def extract_keywords(text):
    doc = nlp(text)
    command = None
    where = None
    where_root = None
    what = None
    for s in doc.sents:
        command = s.root.lemma_
        break
    for token in doc:
        
        if token.head.lemma_ == command:
            if token.dep_ in ["dobj", "pobj"]:
                where_root = token.head
                where = " ".join(
                    [t.text for t in token.lefts] + [token.text])
            if token.dep_ == "prep":
                where_root = token
                where = " ".join(
                    [t.text for t in token.subtree])
        if token.pos_ in what_pos:
            if what is None:
                if token.dep_ in ["dobj", "pobj", "compound", "amod", "nsubj"] and token.head.lemma_ == command or token.head == where_root:
                    what = " ".join(
                        [t.text for t in token.subtree])
                    # what = " ".join(
                    #     [t.text for t in token.lefts] + [token.text] + [t.text for t in token.rights])
    return command, where, what

for text in texts:
    print("Text: ", text)
    command, where, what = extract_keywords(text)
    print("Command: ", command)
    print("Where: ", where)
    print("What: ", what)
    print()


