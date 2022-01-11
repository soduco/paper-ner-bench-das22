#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
# Annotation codes projection to Unicode chars
# Expected input: raw OCR human transcription and annotation (special glyphs w/o public codepoints)
# Output: OCR reference text with projected annotations (before normalization)
# Transform type: (multiple chars) -> (1+ chars), can change string length

# From https://hackmd.io/ocGrovCAQZuKAoGs1zTYIg#Liste-des-glyphes-et-de-leur-repr%C3%A9sentation-en-caract%C3%A8res
ANNOTATION_CODES = [
    # We use unicode private user area when needed (no dedicated unicode char)
    ("::LH::", "\uE001"),  # PUA #1 
    ("::CJ::", "\uE002"),  # PUA #2 
    ("::CM::", "\uE003"),  # PUA #3
    ("::AC::", "\uE004"),  # PUA #4
    ("::PA::", "\uE005"),  # PUA #5
    ("::CP::", "\uE006"),  # PUA #6
    ("::CT::", "\uE007"),  # PUA #7
    ("::CA::", "\uE008"),  # PUA #8
    ("::MM::", "\uE009"),  # PUA #9
    ("::MO::", "\u24c4"),  # → "Ⓞ"
    ("::MA::", "\u24b6"),  # → "Ⓐ"
    ("::MB::", "\u24b7"),  # → "Ⓑ"
    ("::MP::", "\u24c5"),  # → "Ⓟ"
    ("::MV::", "\u24cb"),  # → "Ⓥ"
    # ("M. H.", ""),
    # ("C. F.", ""),
    # ("R. du J. C.", ""),
    # ("B. I.", ""),
    # ("Cit.", ""),
    ("::LG::", "\uE00A"),  # PUA #10
    ("::LP::", "\uE00B"),  # PUA #11
    ("::MH::", "\uE00C"),  # PUA #12 -- /!\ ::LH:: already used
    ("::UJ::", "\uE00D"),  # PUA #13
    ("::UH::", "\uE00E"),  # PUA #14
    ("::UG::", "\uE00F"),  # PUA #15
    ("::UD::", "\uE010"),  # PUA #16
    ("::main::", "\u261E"),  # → "☞"
    ("\U0001F449", "\u261E"),  # "👉" → "☞"  # maybe some extra annotations like this to fix
    ("::SymbET::", "\u002A"),  # → "*"
    ("\u2605", "\u002A"), # "★" → "*"  # maybe some extra annotations like this to fix
    ("::SymbET::", "\u002A"),  # → "*"
    ("::Tvert::", "\u007C"),  # → "|"
    ("::Thoriz::", "\u002D"),  # → "-"
    ("::SymbM::", "\uE011"),  # PUA #17
    ("::NC::", "\uE012"),  # PUA #18
    ("::NA::", "\uE013"),  # PUA #19
    ("::TL::", "\uE014"),  # PUA #20
    ("::TG::", "\uE015"),  # PUA #21
]

def replace_annotation_codes(text: str) -> str:
    res = text
    for fr, to in ANNOTATION_CODES:
        res = res.replace(fr, to)
    return res

# ==============================================================================
# Unicode charset simplification (mainly for XML compatibility)
# Expected input: OCR reference text with projected annotations
# Output: OCR reference text with projected annotations and simplified charset
# Transform type: (1+ chars) -> (0+ chars), can change string length

UNICODE_SIMPLIFICATIONS = [
    # all new lines to \n
    # https://unicode.org/reports/tr13/tr13-9.html
    # ("\u000A", "\u000A"),  # LF == \n
    ("\u000B", "\u000A"),  # VT
    ("\u000C", "\u000A"),  # FF
    ("\u000D", "\u000A"),  # CR == \r
    ("\u000D\u000A", "\u000A"),  # CRLF == \r\n
    ("\u0085", "\u000A"),  # NEL
    ("\u2028", "\u000A"),  # LS
    ("\u2029", "\u000A"),  # PS

    # bars
    ("¦", "|"), # U+00A6 # Commonly interchanged
    ("\u2044", "/"), # ⁄ FRACTION SLASH U+2044

    # quotes and brackets
    ("«", "\""), # U+00AB
    ("»", "\""), # U+00BB
    ("\u2018", "\'"), # ‘ LEFT SINGLE QUOTATION MARK
    ("\u2019", "\'"), # ’ RIGHT SINGLE QUOTATION MARK
    ("\u201A", ","), # ‚ SINGLE LOW-9 QUOTATION MARK
    ("\u201B", "\'"), # ‛ SINGLE HIGH-REVERSED-9 QUOTATION MARK
    ("\u201C", "\""), # “ LEFT DOUBLE QUOTATION MARK
    ("\u201D", "\""), # ” RIGHT DOUBLE QUOTATION MARK
    ("\u201E", "\""), # „ DOUBLE LOW-9 QUOTATION MARK
    ("\u201F", "\""), # ‟ DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    ("‹", "<"), # U+2039
    ("›", ">"), # U+203A
    ("\u2032", "'"), # ′ PRIME
    ("\u2033", "\""), # ′′ DOUBLE PRIME
    ("\u2035", "'"), # ‵ REVERSED PRIME
    ("\u2036", "\""), # ‵‵ REVERSED DOUBLE PRIME

    # Dots
    ("\u2022", "."), # • BULLET
    ("\u2023", "."), # ‣ TRIANGULAR BULLET
    ("\u2024", "."), # . ONE DOT LEADER
    ("\u2025", ".."), # .. TWO DOT LEADER
    ("\u2026", "..."), # ... HORIZONTAL ELLIPSIS
    ("\u2027", "."), # ‧ HYPHENATION POINT

    # Dashes and spaces
    # http://www.unicode.org/charts/PDF/U2000.pdf
    # http://www.unicode.org/charts/PDF/U2E00.pdf
    # dashes
    ("\u2010", "\u002D"), # ‐ HYPHEN
    ("\u2011", "\u002D"), #  NON-BREAKING HYPHEN
    ("\u2012", "\u002D"), # ‒ FIGURE DASH
    ("\u2013", "\u002D"), # – EN DASH
    ("\u2014", "\u002D"), # — EM DASH
    ("\u2015", "\u002D"), # ― HORIZONTAL BAR
    ("\u2E3A", "\u002D"), # ⸺  two-em dash
    ("\u2E3B", "\u002D"), # ⸻ THREE-EM DASH

    # all horiz spaces to space
    ("\u0009", "\u0020"), # HTAB (\t) to SPACE
    ("\u00A0", "\u0020"), # NBSP to SPACE
    ("\u2000", "\u0020"),  # EN QUAD
    ("\u2001", "\u0020"),  # EM QUAD
    ("\u2002", "\u0020"),  # EN SPACE
    ("\u2003", "\u0020"),  # EM SPACE
    ("\u2004", "\u0020"),  # THREE-PER-EM SPACE
    ("\u2005", "\u0020"),  # FOUR-PER-EM SPACE
    ("\u2006", "\u0020"),  # SIX-PER-EM SPACE
    ("\u2007", "\u0020"),  # FIGURE SPACE
    ("\u2008", "\u0020"),  # PUNCTUATION SPACE
    ("\u2009", "\u0020"),  # THIN SPACE
    ("\u200A", "\u0020"),  # HAIR SPACE
    ("\u202F", "\u0020"),  #  NARROW NO-BREAK SPACE

    # Lines
    ("\u2016", "||"),  # ‖ DOUBLE VERTICAL LINE
    ("\u2012", "_"),  #  ̳ DOUBLE LOW LINE

    # Invisible chars
    ("\u200B", ""), # ZERO WIDTH SPACE
    ("\u200C", ""), # ZERO WIDTH NON-JOINER
    ("\u200D", ""), # ZERO WIDTH JOINER
    ("\u200E", ""), # LEFT-TO-RIGHT MARK
    ("\u200F", ""), # RIGHT-TO-LEFT MARK
    ("\u2060", ""), #  WORD JOINER
    ("\u00AD", ""), # SOFT HYPHEN
    ("\ufeff", ""), # ZERO WIDTH NO-BREAK SPACE // Byte Order Mark
]


def simplify_unicode_charset(text: str) -> str:
    res = text
    for fr, to in UNICODE_SIMPLIFICATIONS:
        res = res.replace(fr, to)
    # ???
    #         # Unicode normalization
    #         line_norm = unicodedata.normalize('NFKC', line)
    #         line_norm = unicodedata.normalize('NFC', line)  # TODO
    return res


# ==============================================================================
# Check OCR reference charset
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
            or 0xFB00 <= codepoint <= 0xFB06  # ﬀ, ﬁ, ﬂ, ﬃ, ﬄ, ﬅ, ﬆ (ligatures)
            or codepoint == 0xFFFD  # � REPLACEMENT CHARACTER
            ):
            print(f"Suspect char@{pos:03d}: {char} ({_unichr2str(char)}"
                  f" -- cat.: {unicodedata.category(char)})")
    if not unicodedata.is_normalized('NFKC', text_to_align):
        # Note: circled letter symbols do not seem to comply with NFKC.
        #        indeed: unicodedata.normalize('NFKC', "Ⓞ") ==  "O"
        # Maybe NFC is enough as we want only composed chars.
        print("Warning: string is not normalized (compatibility + composed mode).")
        if not unicodedata.is_normalized('NFC', text_to_align):
            print("Warning: string is not normalized (composed mode).")
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
# Fix manual NER XML

# TODO

# ==============================================================================
# align NER tags

# TODO w/o normalization

# TODO ner charset simplification str.mktrans (accents, uppercase…)

# TODO XML escape
# TODO XML unescape

# ==============================================================================
# OCR eval

# TODO call python function
# TODO OCR equivalences
# TODO case insensitive or not


# ==============================================================================
# read file

# TODO read utf-8 file content

# TODO strip trailing \n
# TODO delete BOM



# 8<----------------------------------




TAG_LIST = ["PER", "LOC", "ACT", "CARDINAL", "FT", "TITRE"]


OCR_EVAL_EQUIVALENCES = [
# ¹ U+00B9 to U+0031  1   <---- use NFKC for this?
# ² U+00B2 to U+0032  2
# ³ U+00B3 to U+0033  3
# µ U+00B5 to U+03BC  μ
    # ("\u00BA", "\u006F"), # º U+00BA to U+006F  o
    # …   U+2026 to U+002E U+002E U+002E    ...  <-------- ???

# ™   U+2122 to U+0054 U+004D   TM
    # ("\uFB00", "ff"),  # Replace ff, fi, fl, ffi, ffl ligatures with separated 
    # ("\uFB01", "fi"),  # chars. before Unicode normalization inserts
    # ("\uFB02", "fl"),  # "200c ZERO WIDTH NON-JOINER"
    # ("\uFB03", "ffi"), # (actually not wrote to UTF-8 output so not done here)
    # ("\uFB04", "ffl"),
# ¼ U+00BC to U+0031 U+002F U+0034  1/4 # done by FRACTION SLASH replace after norm.
# ½ U+00BD to U+0031 U+002F U+0032  1/2 # done by FRACTION SLASH replace after norm.
# ¾ U+00BE to U+0033 U+002F U+0034  3/4 # done by FRACTION SLASH replace after norm.
    # ("Æ", "AE"), # U+00C6  # enable multichar? NO: normalize after OCR
    # ("æ", "ae"), # U+00E6
    # ("Œ", "OE"), # U+0152
    # ("œ", "oe"), # U+0153
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





def simplify_for_alignment(text: str, case_insensitive=True) -> str:
    # Single char "normalizations"
    # tolerance to OCR single char substitutions (case insensitive, no accents, etc.)
    # which DO NOT CHANGE THE MATCHING
    simplified = text
    for fr, to in ALIGNMENT_SIMPLIFICATION_TABLE:
        simplified = simplified.replace(fr, to)
    if case_insensitive:
        simplified = simplified.upper()
    return simplified




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
    A, B = isri_tools.align(a, bs, ' ')  # ␣
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
