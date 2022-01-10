import nltk
import spacy
from spacy.tokens import DocBin, Doc
from xml.dom.minidom import parseString
from spacy.training import tags_to_entities

# =============================================================================
# region ~ Data conversion utils for Spacy

nlp = spacy.blank("fr")


def create_spacy_dataset(entries):
    db = DocBin()

    for entry in entries:
        # Makes the entry a valid XML string, then parses it to a DOM.
        xml = parseString(f"<xml>{entry}</xml>").getElementsByTagName("xml")[0]

        tags = []
        words = []
        text = ""
        for el in xml.childNodes:
            el_is_txt = el.nodeName == "#text"
            span = el.nodeValue if el_is_txt else el.childNodes[0].nodeValue
            text += span

            if span.isspace():
                continue

            span_toks = nltk.word_tokenize(span, language="fr", preserve_line=True)

            # BILUO as pivot format
            nertag = el.nodeName.upper()
            span_biluo = []
            if not el_is_txt:
                if len(span_toks) == 1:
                    span_biluo = [f"U-{nertag}"]
                elif len(span_toks) > 1:
                    span_biluo = (
                        [f"B-{nertag}"]
                        + [f"I-{nertag}"] * max(len(span_toks) - 2, 0)
                        + [f"L-{nertag}"]
                    )
                else:
                    raise ValueError(f"Empty inner doc in {span}: {entry}")
                tags.extend(span_biluo)
            else:
                tags.extend(["O"] * len(span_toks))

            words.extend(span_toks)

        doc = Doc(nlp.vocab, words=words)
        ents = tags_to_entities(tags)
        doc.ents = [
            spacy.tokens.Span(doc, start=s, end=e + 1, label=lbl)
            for (lbl, s, e) in ents
        ]
        db.add(doc)

    return db
