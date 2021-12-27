
import argparse
import json
import sys
import re

# "text_ocr": "##Castries (Mis de) C.## µµ::LH::µµ, ££mar. de camp££, @@Varennes@@, $$22$$.\n",
char_map = {
    "#": "PER",
    "µ": "TITRE",
    "£": "ACT",
    "@": "LOC",
    r"\$": "CARD",
#    "%": "FT",  # ???
}

def parse_ascii_annot(text_orig):
    """text_orig: str -> (text_ocr: str, ner_xml: str)"""
    # For each ASCII tag, we remove the annotation from text_ocr and create xml annot
    text_ocr = text_orig  # str are immutables
    ner_xml = text_orig  # str are immutables
    for char, tag in char_map.items():
        pattern = f"{char}{char}([^{char}]+){char}{char}"
        repl_ocr = r"\1"
        repl_xml = f"<{tag}>" + r"\1" + f"</{tag}>"
        # Warning we must accept newlines within patterns
        text_ocr = re.sub(pattern, repl_ocr, text_ocr, flags=re.DOTALL)
        ner_xml = re.sub(pattern, repl_xml, ner_xml, flags=re.DOTALL)
    return text_ocr, ner_xml

def main():
    parser = argparse.ArgumentParser(description='Convert ASCII tag format to XML tag format.')
    parser.add_argument('input_file',
                        help='Path to input file.')
    parser.add_argument('output_file',
                        help='Path to output file.')

    args = parser.parse_args()

    print(f"Processing {args.input_file} -> {args.output_file}")

    # We will update this variable in place
    annotations = None
    with open(args.input_file, 'r') as input_file:
        annotations = json.load(input_file)
    
    if not isinstance(annotations, list):
        raise ValueError("Cannot read a list of annotations from the input.")
    
    # Convert content
    for annot in annotations:
        if not ("type" in annot and annot["type"] == "ENTRY"):
            continue
        # updated checked
        annot["checked"] = False
        # update text_ocr and ner_xml fields
        if "text_ocr" not in annot:
            continue
        text_ocr, ner_xml = parse_ascii_annot(annot["text_ocr"])
        annot["text_ocr"] = text_ocr
        annot["ner_xml"] = ner_xml
    
    # Write output
    with open(args.output_file, 'w') as output_file:
        json.dump(annotations, output_file)


if __name__ == "__main__":
    rc = main()
    if rc != None:
        sys.exit(rc)
