#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This program synchronizes NER tagging from reference text to noisy predicted
text (OCR output).

Sample usage:
    ner_sync_ref.py reference_utf8.xml [predicted_utf8.xml | predicted_utf8.txt] [alignedref_utf8.xml]

Input restrictions:
    - Encoding MUST BE UTF-8.
    - End of line MUST BE either CRLF (Windows), CR (old Macintosh) or LF (OSX,
      *nix).

Output post-conditions:
    - Encoding WILL BE UTF-8.
    - End of line WILL BE LF.
"""

# Reused code from https://github.com/SmartDOC-MOC/moc_normalization

# ==============================================================================
# Imports
from collections import Counter
import io
import logging
import argparse
import sys
import unicodedata
from warnings import warn
from typing import List, Tuple

import regex
import numpy as np
import isri_tools

# ==============================================================================
# Logging
logger = logging.getLogger(__name__)

# ==============================================================================
# Constants
PROG_VERSION = "1.0"
PROG_DESCR = (
    "Synchronizes NER tagging from reference text to noisy predicted text (OCR output)."
    f" Version: {PROG_VERSION}"
    )
PROG_NAME = "ner_sync_ref"

ERRCODE_OK = 0
ERRCODE_NOFILE = 10
ERRCODE_EXTRACHAR = 50

# FIXME extract this to a common config file (if we have multiple scripts)
# FIXME add cyrillic as Pero OCR can output content in this script (actually add entire scripts)
#       OR project extra char to unused char in GT (Unicode PUA? or U+FFFD � REPLACEMENT CHARACTER)
# ALLOWED_INPUT = (
#     """ !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abc"""
#     """defghijklmnopqrstuvwxyz{|}~ ¡¢£¤¥¦§¨©ª«¬­®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉ"""
#     """ÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿŒœŠšŸŽžƒˆ˜–—‘’"""
#     """‚“”„†‡•…‰‹›€™ﬁﬂﬀﬃﬄ"""
# # Horizontal tab added separately for convenience
#     "\u0009"
# # additions due to Unicode normalization:
# # - 0308 COMBINING DIAERESIS
# # - 0301 COMBINING ACUTE ACCENT
# # - 03BC GREEK SMALL LETTER MU
# # - 0327 COMBINING CEDILLA
# # - 0303 COMBINING TILDE
# # - 0304 COMBINING MACRON
# # - 2044 FRACTION SLASH
#     "\u0308\u0301\u03BC\u0327\u0303\u0304\u2044"
# # ZERO WIDTH NO-BREAK SPACE
#     "\ufeff"
#     )

TAG_LIST = ["PER", "LOC", "ACT", "CARDINAL", "FT", "TITRE"]

# From https://hackmd.io/ocGrovCAQZuKAoGs1zTYIg#Liste-des-glyphes-et-de-leur-repr%C3%A9sentation-en-caract%C3%A8res
ANNOT_CODE_MAP = {  # We use unicode private user area when needed (no dedicated unicode char)
    "::LH::": "\uE001",  # PUA #1
    "::CJ::": "\uE002",  # PUA #2
    "::CM::": "\uE003",  # PUA #3
    "::AC::": "\uE004",  # PUA #4
    "::PA::": "\uE005",  # PUA #5
    "::CP::": "\uE006",  # PUA #6
    "::CT::": "\uE007",  # PUA #7
    "::CA::": "\uE008",  # PUA #8
    "::MM::": "\uE009",  # PUA #9
    "::MO::": "\u24c4",  # "Ⓞ"
    "::MA::": "\u24b6",  # "Ⓐ"
    "::MB::": "\u24b7",  # "Ⓑ"
    "::MP::": "\u24c5",  # "Ⓟ"
    "::MV::": "\u24cb",  # "Ⓥ"
    # "M. H.": "",
    # "C. F.": "",
    # "R. du J. C.": "",
    # "B. I.": "",
    # "Cit.": "",
    "::LG::": "\uE00A",
    "::LP::": "\uE00B",
    "::MH::": "\uE00C",  # ::LH:: already used
    "::UJ::": "\uE00D",
    "::UH::": "\uE00E",
    "::UG::": "\uE00F",
    "::UD::": "\uE010",
    "::main::": "\u261E",  # ☞
    "::SymbET::": "*",
    "\u2605": "*", # ★
    "::SymbET::": "*",
    "::Tvert::": "|",
    "::Thoriz::": "-",
    "\u2013": "-", # –
    "\u2014": "-", # —
    "::SymbM::": "\uE011",
    "::NC::": "\uE012",
    "::NA::": "\uE013",
    "::TL::": "\uE014",
    "::TG::": "\uE015",
}

SIMPLIFICATIONS = [
    ("\u0009", " "), # HTAB to SPACE
    ("\u00A0", " "), # NBSP to SPACE
    ("¦", "|"), # U+00A6 # Commonly interchanged
# ¨ U+00A8 to U+0020 U+0308    ̈
# ª U+00AA to U+0061  a
    ("«", "\""), # U+00AB
    # ("\u00AD", ""), # remove SOFT HYPHEN
# ¯ U+00AF to U+0020 U+0304    ̄
# ² U+00B2 to U+0032  2
# ³ U+00B3 to U+0033  3
# ´ U+00B4 to U+0020 U+0301    ́
# µ U+00B5 to U+03BC  μ
# ¸ U+00B8 to U+0020 U+0327    ̧
# ¹ U+00B9 to U+0031  1
    ("\u00BA", "\u006F"), # º U+00BA to U+006F  o
    ("»", "\""), # U+00BB
# ¼ U+00BC to U+0031 U+002F U+0034  1/4 # done by FRACTION SLASH replace after norm.
# ½ U+00BD to U+0031 U+002F U+0032  1/2 # done by FRACTION SLASH replace after norm.
# ¾ U+00BE to U+0033 U+002F U+0034  3/4 # done by FRACTION SLASH replace after norm.
    # ("Æ", "AE"), # U+00C6  # enable multichar? NO: normalize after OCR
    # ("æ", "ae"), # U+00E6
    # ("Œ", "OE"), # U+0152
    # ("œ", "oe"), # U+0153
#˜   U+02DC to U+0020 U+0303    ̃
    ("–", "-"), # U+2013
    ("—", "-"), # U+2014
    ("‘", "\'"), # U+2018
    ("’", "\'"), # U+2019
    ("‚", "\'"), # U+201A
    ("“", "\""), # U+201C
    ("”", "\""), # U+201D
    ("„", "\""), # U+201E
# …   U+2026 to U+002E U+002E U+002E    ...
    ("‹", "\'"), # U+2039
    ("›", "\'"), # U+203A
# ™   U+2122 to U+0054 U+004D   TM
    # ("\uFB00", "ff"),  # Replace ff, fi, fl, ffi, ffl ligatures with separated 
    # ("\uFB01", "fi"),  # chars. before Unicode normalization inserts
    # ("\uFB02", "fl"),  # "200c ZERO WIDTH NON-JOINER"
    # ("\uFB03", "ffi"), # (actually not wrote to UTF-8 output so not done here)
    # ("\uFB04", "ffl"),
    ("⁄", "/"), # FRACTION SLASH U+2044
    # ("\ufeff", ""), # ZERO WIDTH NO-BREAK SPACE // Byte Order Mark
    # TODO remove accents?
    ("À", "A"),
    ("Ç", "C"),
    ("É", "E"),
    ("È", "E"),
    ("Ê", "E"),
    ("Ë", "E"),
    ("Î", "I"),
    ("Ï", "I"),
    ("Ô", "O"),
    ("Ö", "O"),
    ("Û", "U"),
    ("Ü", "U"),
    ("à", "a"),
    ("ç", "c"),
    ("é", "e"),
    ("è", "e"),
    ("ê", "e"),
    ("ë", "e"),
    ("î", "i"),
    ("ï", "i"),
    ("ô", "o"),
    ("ö", "o"),
    ("û", "u"),
    ("ü", "u"),
    ("@", "a"),
    ]

# ==============================================================================
def check_xml(ner_xml: str, taglist: List[str]) -> bool:
    cbegin = { tag : ner_xml.count(f"<{tag}>") for tag in taglist }
    cend = { tag : ner_xml.count(f"</{tag}>") for tag in taglist }
    if cbegin != cend:
        warn(f"The string '{ner_xml}' has unbalanced tags.")
        print("Opening:", cbegin)
        print("Closing:", cend)
        return False
    # TODO check for unescaped XML entities, isolated "&", etc.
    return True


def chop_on_tags(ner_xml: str, tag_list: List[str]) -> Tuple[str,str]:
    """Chops the input string on XML tags.

    Example:
    ```
    ner_xml = "<PER>Anthony</PER>, fab. du <ACT>pêche</ACT>, <LOC>Châtelet</LOC>"
    chunks, tags = chop_on_tags(ner_xml, tag_list=["PER", "ACT", "LOC"])
    print(chunks)
    # > ['', 'Anthony', ', fab. du ', 'pêche', ', ', 'Châtelet', '']
    print(tags)
    # > ['<PER>', '</PER>', '<ACT>', '</ACT>', '<LOC>', '</LOC>']
    ```
    Args:
        ner_xml (str): XML string to chop.
        tag_list (List[str]): Tags to use.

    Returns:
        Tuple[str,str]: pair of (texts, tags) where:
            - texts is the list of strings between beginning of line, tags and end of line
            - tags is the list of opening and closing tags in their order of appearance
    """
    chunks = regex.split("(</?\L<tag>>)", ner_xml, tag=tag_list) # TODO ignore case for tags?
    A_chunks = chunks[0::2]
    A_tags = [] if len(chunks) < 2 else chunks[1::2]
    return A_chunks, A_tags


def xml_unescape(text: str) -> str:
    return text


def replace_annotation_codes(text: str) -> str:
    res = text
    for code in ANNOT_CODE_MAP:
        res = res.replace(code, ANNOT_CODE_MAP[code])
    return res


def simplify_for_alignment(text: str, case_insensitive=True) -> str:
    # Single char "normalizations"
    # tolerance to OCR single char substitutions (case insensitive, no accents, etc.)
    # which DO NOT CHANGE THE MATCHING
    simplified = text
    for fr, to in SIMPLIFICATIONS:
        simplified = simplified.replace(fr, to)
    if case_insensitive:
        simplified = simplified.upper()
    return simplified


def check_alignment_charset(text_to_align: str, dump_charset: bool) -> bool:
    """Check the string passed as parameter contains only acceptable
    unicode code points for the alignment.

    The alignment tool (coded before Unicode 2.0 in 1996) currently requires
    that code points are 16 bits wide at most.
    We also add the constraint that we do not want UTF-16 surrogates nor non-characters.

    Args:
        text_to_align (str): text to check before alignment
        dump_charset (bool): whether to dump the charset or not (debug)

    Returns:
        bool: Return True if the text has a valid charset, False otherwise.
    """
    errors = 0
    for pos, char in enumerate(text_to_align):
        codepoint = ord(char)
        if (
            codepoint in (0x0000, 0xFFFE, 0xFFFF)  # NULL, non chars
            or 0xD800 <= codepoint <= 0xDFFF  # surrogates
            or codepoint > 0xFFFF):  # large codepoints
            errors += 1
            print(f"Invalid char@{pos:03d}: {char} ({_unichr2str(char)})")
        elif not (
            codepoint in (ord("\n"), ord("\r"))
            or 0x0020 <= codepoint <= 0x007E  # Basic_Latin
            or 0x00A0 <= codepoint <= 0x00FF  # Latin-1_Supplement
            or 0x0100 <= codepoint <= 0x017F  # Latin Extended-A
            or 0x0200 <= codepoint <= 0x026F  # General Punctuation
            or codepoint in (0x24b6, 0x24b7, 0x24c4, 0x24c5, 0x24cb) # Ⓐ, Ⓑ, Ⓞ, Ⓟ, Ⓥ: Enclosed Alphanumerics (2460–24FF)
            or codepoint in (0x261E, )  # 0x261E:☞ (0x2605: ★): Miscellaneous Symbols (2600–26FF)
            or codepoint in (0x2709, )  # ✉: Dingbats (2700–27BF)
            or 0xE000 <= codepoint <= 0xE0FF  # first 255 chars of Private Use Area (E000–F8FF)
            or codepoint == 0xFFFD  # � REPLACEMENT CHARACTER
            ):
            print(f"Suspicious char@{pos:03d}: {char} ({_unichr2str(char)}"
                  f" -- cat.: {unicodedata.category(char)})")
    if not unicodedata.is_normalized('NFKC', text_to_align):
        print("Warning: string is not normalized (compatibility + composed mode).")
    if dump_charset:
        print("Charset for string:")
        print(f"\t\"{text_to_align}\"")
        charset = Counter(text_to_align)
        print("\t   # | Char")
        print("\t ----+---------------------")
        for char, count in charset.most_common():
            char_str = char if char != "\n" else "<NL>"
            print(f"\t {count:3d} | {char_str} ({_unichr2str(char)} -- cat.: {unicodedata.category(char)})")
        print("")
    return errors == 0


def _unichr2str(unichar: str) -> str:
    return unicodedata.name(unichar, repr(unichar))

# ==============================================================================
# FIXME extract this to separate tool which will be run on all files/strings
def project_to_simple_charset(text: str) -> str:
    pass
    # with io.open(args.output, "wt", encoding="UTF-8", newline='',
    #             errors="strict") as file_output:
    # # output lines are utf-8-encoded and have LF EOL
    # line_no = 0
    # with io.open(args.input, "rt", encoding="UTF-8", newline=None,
    #                 errors="strict") as file_input:
    #     # input lines are Unicode code point sequences with LF-normalized EOL
    #     for line in file_input:
    #         line_no += 1

    #         # Check input
    #         extra_chars = []
    #         char_no = 0
    #         for char in line:
    #             char_no += 1
    #             if char not in charset:
    #                 extra_chars.append((char, char_no))
    #                 err_count += 1
    #             # logger.debug("\tl:%03d c:%03d %04x %s" 
    #             #     % (line_no, char_no, ord(char), _unichr2str(char)))
    #         if extra_chars:
    #             logger.error("Got %d illegal character(s) in line %d : " 
    #                             % (len(extra_chars), line_no))
    #             for i in range(min(CHAR_ERR_LIM, len(extra_chars))):
    #                 char, pos = extra_chars[i]
    #                 logger.error("\tl:%03d c:%03d %s" 
    #                                 % (line_no, pos, _unichr2str(char)))
    #             if len(extra_chars) > CHAR_ERR_LIM:
    #                 logger.error("\t ... and %d other(s)." 
    #                                 % (len(extra_chars) - CHAR_ERR_LIM))

    #         # Unicode normalization
    #         line_norm = unicodedata.normalize('NFKC', line)
            
    #         # Perform custom translations
    #         line_tr = _transform(line_norm)

    #         # Output new line
    #         file_output.write(line_tr) 


def add_tags_prediction(ner_xml: str, text_ocr: str, debug=False):

    # 1. Process ner_xml
    ## 1.0. Sanity check
    _xml_a_correct = check_xml(ner_xml, taglist=TAG_LIST)
    ## 1.1. Chunking
    A_chunks, A_tags = chop_on_tags(ner_xml, tag_list=TAG_LIST)
    ## 1.2. Replace XML entities
    # We don't do it here as the GUI tools doesn't do it.
    A_chunks = list(map(xml_unescape, A_chunks))
    ## 1.3. Replace annotation codes (e.g. "::LH::") with (private if needed) unicode chars
    A_chunks = list(map(replace_annotation_codes, A_chunks))
    # TODO
    ## 1.4. Simplify the string to ease alignment (e.g. "é"->"e")
    A_chunks = list(map(simplify_for_alignment, A_chunks))
    # TODO
    # Note: single char projections for now, requires to store complex transforms
    # in other string otherwise. Plus charset simplification should be performed earlier.
    ## 1.5. Build string ready for alignment
    a = "".join(A_chunks)
    _charset_a_correct = check_alignment_charset(a, dump_charset=debug)
    # => a is now ready for alignment

    # B. Process text_ocr
    # Assume we already have a normalized unicode text without any XML content (tags or entities)
    # 2.1 Simplify the string to ease alignment (e.g. "é"->"e")
    b = text_ocr  # We keep b and align on a simplified version of it
    bs = simplify_for_alignment(text_ocr)
    if len(b) != len(bs):
        raise RuntimeError("simplify_for_alignment changed string len. Cannot align simplified string.")
    _charset_b_correct = check_alignment_charset(bs, dump_charset=debug)


    # 3. Align
    A, B = isri_tools.align(a, bs, ' ')
    # print(A)
    # print(B)
    A, B = isri_tools.get_align_map(a, bs)

    # 4. 
    pos_tags = np.cumsum([len(x) for x in A_chunks[:-1]])

    # 5. Reprojet b on the alignment string
    n = max(np.max(A), np.max(B)) + 1
    chr_list = [ '' for i in range(n + 1)]
    for k, c in zip(B, b):
        chr_list[k] = c


    # 6. Add tags on the alignment string
    # FIXME here it would be safer to cut at insertion position, escape XML entities, then recombine
    for p, tag in zip(reversed(pos_tags), reversed(A_tags)):
        if tag.startswith("</"):
            # print(p, a[p-1], chr_list[A[p-1] + 1])
            chr_list.insert(A[p-1]+1, tag)
        else:
            # print(p, a[p], chr_list[A[p]])
            chr_list.insert(A[p], tag)

    return "".join(chr_list)


def remove_xml_tags_and_entities(xml: str) -> str:
    # We reuse existing functions here
    chunks, _tags = chop_on_tags(xml, tag_list=TAG_LIST)
    chunks = list(map(xml_unescape, chunks))
    return "".join(chunks)

def remove_unicode_bom(text: str) -> str:
    if text.startswith("\uFEFF"): # ZERO WIDTH NO-BREAK SPACE // Byte Order Mark
        return text[1:]
    return text

# ==============================================================================
# Utility private functions
def _dump_args(args, logger=logger):
    logger.debug("Arguments:")
    for (k, v) in args.__dict__.items():
        logger.debug("    %-20s = %s", k, v)

_DBGLINELEN = 80
_DBGSEP = "-"*_DBGLINELEN

def _program_header(logger, prog_name, prog_version):
    logger.debug(_DBGSEP)
    dbg_head = "%s - v. %s" % (prog_name, prog_version)
    dbg_head_pre = " " * (max(0, (_DBGLINELEN - len(dbg_head)))//2)
    logger.debug(dbg_head_pre + dbg_head)

def _init_logger(logger, debug=False):
    fmt="%(module)-9s %(levelname)-7s: %(message)s"
    formatter = logging.Formatter(fmt)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    level = logging.INFO
    if debug:
        level = logging.DEBUG
    logger.setLevel(level)

# ==============================================================================
# Main function
def main():
    # Option parsing
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=PROG_DESCR,
        epilog=__doc__)
    parser.add_argument('-d', '--debug',
        action="store_true",
        help="Activate debug output.")
    parser.add_argument('reference', 
        help='Input reference file with UTF-8 encoding and XML tags.')
    parser.add_argument('predicted', 
        help='Input prediction file with UTF-8 encoding (and optionally XML tags).')
    # parser.add_argument('output',  #  TODO add optional arg (write to stdout if absent)
    #     help="Path to synchronized output file (predicted text with reference tags synchronized).")
    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # Logger activation
    _init_logger(logger)
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # --------------------------------------------------------------------------
    # Output log header
    _program_header(logger, PROG_NAME, PROG_VERSION)
    logger.debug(_DBGSEP)
    _dump_args(args, logger)
    logger.debug(_DBGSEP)

    # --------------------------------------------------------------------------
    logger.debug("--- Process started. ---")

    # read reference
    ner_xml = None
    with io.open(args.reference, "rt", encoding="UTF-8", newline=None,
                 errors="strict") as file_input:
        ner_xml = file_input.read()
    # remove leading byte order mark
    ner_xml = remove_unicode_bom(ner_xml)
    # remove trailing newlines?
    ner_xml = ner_xml.rstrip("\n")
    # TODO trim whitespaces?

    # read OCR prediction
    ocr_text = None
    with io.open(args.predicted, "rt", encoding="UTF-8", newline=None,
                 errors="strict") as file_input:
        ocr_text = file_input.read()
    # remove leading byte order mark
    ocr_text = remove_unicode_bom(ocr_text)
    # remove trailing newlines?
    ocr_text = ocr_text.rstrip("\n")
    # TODO trim whitespaces?

    if args.predicted.endswith(".xml"):
        logger.warning(f"File {args.predicted} seem to be an XML file. Removing XML tags and entities.")
        ocr_text = remove_xml_tags_and_entities(ocr_text)

    # Main algorithm
    res = add_tags_prediction(ner_xml, ocr_text, debug=args.debug)

    # FIXME Stupid output for now
    print(res)  # output result to stdout

    logger.debug("--- Process complete. ---")
    # --------------------------------------------------------------------------

    ret_code = ERRCODE_OK
    # if err_count > 0:
    #     logger.error(_DBGSEP)
    #     logger.error("Input file contains %d illegal characters.", err_count)
    #     ret_code = ERRCODE_EXTRACHAR
    # else:
    #     logger.debug("Input file contains only legal characters.")

    logger.debug("Clean exit.")
    logger.debug(_DBGSEP)
    return ret_code
    # --------------------------------------------------------------------------

# ==============================================================================
# Entry point
if __name__ == "__main__":
    sys.exit(main())
