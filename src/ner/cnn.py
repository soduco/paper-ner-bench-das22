import spacy
from spacy.tokens import DocBin, Doc
from xml.dom.minidom import parseString
from spacy.training import biluo_tags_to_offsets

nlp = spacy.blank("fr")


def create_spacy_dataset(entries):
    db = DocBin()

    for entry in entries:
        # Makes the entry a valid XML string, then parses it to a DOM.
        entry_xml = f"<x>{entry}</x>"
        x = parseString(entry_xml).getElementsByTagName("x")[0]

        tags = []
        tokens = []
        text = ""
        for el in x.childNodes:
            el_is_txt = el.nodeName == "#text"

            span = el.nodeValue if el_is_txt else el.childNodes[0].nodeValue
            text += span
            span = (
                span.strip()
            )  # Remove leading whitespaces or they will be kept as tokens

            inner_doc = nlp.make_doc(span)
            nertag = el.nodeName.upper()

            # To BILUO
            inner_tags = []
            if not el_is_txt:
                if len(inner_doc) == 1:
                    inner_tags = [f"U-{nertag}"]
                elif len(inner_doc) > 1:
                    inner_tags = (
                        [f"B-{nertag}"]
                        + [f"I-{nertag}"] * max(len(inner_doc) - 2, 0)
                        + [f"L-{nertag}"]
                    )
                else:
                    raise ValueError(f"Empty inner doc in {span}: {entry}")
                tags.extend(inner_tags)
            else:
                tags.extend(["O"] * len(inner_doc))

            tokens.extend([t.text for t in inner_doc])

        words, spaces = spacy.util.get_words_and_spaces(tokens, text)
        doc = Doc(nlp.vocab, words=words, spaces=spaces)

        offsets = biluo_tags_to_offsets(doc, tags)
        doc.ents = [doc.char_span(start, end, label=lbl) for start, end, lbl in offsets]
        db.add(doc)

    return db
