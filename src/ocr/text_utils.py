#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
# Imports
from collections import Counter
import io
import logging
from typing import List, Tuple
import unicodedata
from warnings import warn

import isri_tools
import numpy as np
import regex

# ==============================================================================
# Logging
logger = logging.getLogger(__name__)



# ==============================================================================
# Some constants
TAG_LIST = ["PER", "LOC", "ACT", "CARDINAL", "FT", "TITRE"]


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
    ("::SymbET::", "\u002A"),  # → "*" (note that \u002A can be represented with 5 or 6 branches in Unicode)
    ("\u2605", "\u002A"), # "★" → "*"  # maybe some extra annotations like this to fix
    ("::SymbET::", "\u002A"),  # → "*" (note that \u002A can be represented with 5 or 6 branches in Unicode)
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
# https://www.w3.org/TR/unicode-xml/#Suitable
# Expected input: OCR reference text with projected annotations
# Output: OCR reference text with projected annotations and simplified charset
# General transform: (1+ chars) -> (1+ chars) (no deletion for debug purpose)

# Transform type: (1 char) -> (1+ char), can change string length
UNICODE_SIMPLIFICATIONS_SINGLE_CHAR = str.maketrans({
    # all new lines to \n
    # https://unicode.org/reports/tr13/tr13-9.html
    # "\u000A": "\u000A",  # LF == \n
    "\u000B": "\u000A",  # VT
    "\u000C": "\u000A",  # FF
    "\u000D": "\u000A",  # CR == \r
    # ("\u000D\u000A", "\u000A"),  # CRLF == \r\n  # move to general substitutions for performance
    "\u0085": "\u000A",  # NEL
    "\u2028": "\u000A",  # LS
    "\u2029": "\u000A",  # PS

    # bars
    "¦": "|", # U+00A6 # Commonly interchanged
    "\u2044": "/", # ⁄ FRACTION SLASH U+2044

    # quotes and brackets
    "«": "\"", # U+00AB
    "»": "\"", # U+00BB
    "\u2018": "\'", # ‘ LEFT SINGLE QUOTATION MARK
    "\u2019": "\'", # ’ RIGHT SINGLE QUOTATION MARK
    "\u201A": ",", # ‚ SINGLE LOW-9 QUOTATION MARK
    "\u201B": "\'", # ‛ SINGLE HIGH-REVERSED-9 QUOTATION MARK
    "\u201C": "\"", # “ LEFT DOUBLE QUOTATION MARK
    "\u201D": "\"", # ” RIGHT DOUBLE QUOTATION MARK
    "\u201E": "\"", # „ DOUBLE LOW-9 QUOTATION MARK
    "\u201F": "\"", # ‟ DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    "\u2032": "'", # ′ PRIME
    "\u2033": "\"", # ′′ DOUBLE PRIME
    "\u2035": "'", # ‵ REVERSED PRIME
    "\u2036": "\"", # ‵‵ REVERSED DOUBLE PRIME
    "‹": "<", # U+2039
    "›": ">", # U+203A

    # Dots
    "\u2022": ".", # • BULLET
    "\u2023": ".", # ‣ TRIANGULAR BULLET
    "\u2024": ".", # . ONE DOT LEADER
    "\u2025": "..", # .. TWO DOT LEADER
    "\u2026": "...", # ... HORIZONTAL ELLIPSIS
    "\u2027": ".", # ‧ HYPHENATION POINT

    # Dashes and spaces
    # http://www.unicode.org/charts/PDF/U2000.pdf
    # http://www.unicode.org/charts/PDF/U2E00.pdf
    # dashes
    "\u2010": "\u002D", # ‐ HYPHEN
    "\u2011": "\u002D", #  NON-BREAKING HYPHEN
    "\u2012": "\u002D", # ‒ FIGURE DASH
    "\u2013": "\u002D", # – EN DASH
    # "\u2014": "\u002D", # — EM DASH  # /!\ kept because we have special chars like this
    "\u2015": "\u002D", # ― HORIZONTAL BAR
    "\u2E3A": "\u002D", # ⸺  two-em dash
    "\u2E3B": "\u002D", # ⸻ THREE-EM DASH

    # all horiz spaces to plain space
    "\u0009": "\u0020", # HTAB (\t)
    "\u00A0": "\u0020", # NBSP
    "\u2000": "\u0020",  # EN QUAD
    "\u2001": "\u0020",  # EM QUAD
    "\u2002": "\u0020",  # EN SPACE
    "\u2003": "\u0020",  # EM SPACE
    "\u2004": "\u0020",  # THREE-PER-EM SPACE
    "\u2005": "\u0020",  # FOUR-PER-EM SPACE
    "\u2006": "\u0020",  # SIX-PER-EM SPACE
    "\u2007": "\u0020",  # FIGURE SPACE
    "\u2008": "\u0020",  # PUNCTUATION SPACE
    "\u2009": "\u0020",  # THIN SPACE
    "\u200A": "\u0020",  # HAIR SPACE
    "\u202F": "\u0020",  # NARROW NO-BREAK SPACE
    "\u205F": "\u0020",  # MEDIUM MATHEMATICAL SPACE

    # Lines H/V
    "\u2016": "||",  # ‖ DOUBLE VERTICAL LINE
    "\u2017": "_",  #  ̳ DOUBLE LOW LINE

    # Invisible chars and layout control → 0xFFFD � REPLACEMENT CHARACTER
    # We replace rather than deleting to enable debugging
    "\u00AD": "\uFFFD",  # SOFT HYPHEN
    "\u200B": "\uFFFD",  # ZERO WIDTH SPACE
    "\u200C": "\uFFFD",  # ZERO WIDTH NON-JOINER
    "\u200D": "\uFFFD",  # ZERO WIDTH JOINER
    "\u200E": "\uFFFD",  # LEFT-TO-RIGHT MARK
    "\u200F": "\uFFFD",  # RIGHT-TO-LEFT MARK
    "\u202A": "\uFFFD",  # LEFT-TO-RIGHT EMBEDDING
    "\u202B": "\uFFFD",  # RIGHT-TO-LEFT EMBEDDING
    "\u202C": "\uFFFD",  # POP DIRECTIONAL FORMATTING
    "\u202D": "\uFFFD",  # LEFT-TO-RIGHT OVERRIDE
    "\u202E": "\uFFFD",  # RIGHT-TO-LEFT OVERRIDE
    "\u2060": "\uFFFD",  # WORD JOINER
    "\u2061": "\uFFFD",  # FUNCTION APPLICATION
    "\u2062": "\uFFFD",  # INVISIBLE TIMES
    "\u2063": "\uFFFD",  # INVISIBLE SEPARATOR
    "\u2064": "\uFFFD",  # INVISIBLE PLUS
    "\u2066": "\uFFFD",  # LEFT-TO-RIGHT ISOLATE
    "\u2067": "\uFFFD",  # RIGHT-TO-LEFT ISOLATE
    "\u2068": "\uFFFD",  # FIRST STRONG ISOLATE
    "\u2069": "\uFFFD",  # POP DIRECTIONAL ISOLATE

    # Deprecated chars
    "\u206A": "\uFFFD",  # INHIBIT SYMMETRIC SWAPPING
    "\u206B": "\uFFFD",  # ACTIVATE SYMMETRIC SWAPPING
    "\u206C": "\uFFFD",  # INHIBIT ARABIC FORM SHAPING
    "\u206D": "\uFFFD",  # ACTIVATE ARABIC FORM SHAPING
    "\u206E": "\uFFFD",  # NATIONAL DIGIT SHAPES
    "\u206F": "\uFFFD",  # NOMINAL DIGIT SHAPES

    # Interlinear annotation
    "\uFFF9": "\uFFFD",  # INTERLINEAR ANNOTATION ANCHOR
    "\uFFFA": "\uFFFD",  # INTERLINEAR ANNOTATION SEPARATOR
    "\uFFFB": "\uFFFD",  # INTERLINEAR ANNOTATION TERMINATOR

    # Replacement characters
    "\uFFFC": "\uFFFD",  # OBJECT REPLACEMENT CHARACTER
    # Keep FFFD � REPLACEMENT CHARACTER
    
    # Byte order mark
    "\ufeff": "\uFFFD", # ZERO WIDTH NO-BREAK SPACE
})


def simplify_unicode_charset(text: str) -> str:
    # Unicode normalization to composed characters (remove combining chars)
    # We do not use 'NFKC' because the substitutions may not be the ones we want.
    res = unicodedata.normalize('NFC', text)

    # Replace CRLR to LF (must be first)
    res = res.replace("\u000D\u000A", "\u000A")

    # Apply all our 1-char substitutions efficiently
    res = res.translate(UNICODE_SIMPLIFICATIONS_SINGLE_CHAR)

    # At this point: 
    # - all new line control chars are mapped to "\n" (\u000A)
    # - all horizontal space control chars are mapped to " " (\u0020)

    # Replace control chars (but \n) with 0xFFFD � REPLACEMENT CHARACTER
    # (XML compatibility)
    # https://www.unicode.org/charts/PDF/U0000.pdf
    # https://www.unicode.org/charts/PDF/U0080.pdf
    res = regex.sub("[\u0000-\u0009\u000B-\u001F\u007F]", "\uFFFD", res)  # C0
    res = regex.sub("[\u0080-\u009F]", "\uFFFD", res)  # C1

    # Replace Variation Selectors with 0xFFFD � REPLACEMENT CHARACTER
    # https://www.unicode.org/charts/PDF/UFE00.pdf
    res = regex.sub("[\uFE00-\uFE0F]", "\uFFFD", res)

    # Replace surrogate UTF-16 chars with 0xFFFD � REPLACEMENT CHARACTER
    # https://www.unicode.org/charts/PDF/UD800.pdf
    # https://www.unicode.org/charts/PDF/UDC00.pdf
    res = regex.sub("[\uD800-\uDFFF]", "\uFFFD", res)

    # Replace non characters from Unicode BMP with 0xFFFD � REPLACEMENT CHARACTER
    # https://www.unicode.org/charts/PDF/UFB50.pdf
    # https://www.unicode.org/charts/PDF/UFFF0.pdf
    # FDD0-FDEF + FFFE + FFFF -> 0xFFFD � REPLACEMENT CHARACTER
    res = regex.sub("[\uFDD0-\uFDEF\uFFFE\uFFFF]", "\uFFFD", res)

    # Note: we could also target unassigned codes but charset control
    #       should be used instead.
    
    # Replace codes > 0xFFFF (chars not in Unicode BMP)
    res = regex.sub("[^\u0000-\uFFFF]", "\uFFFD", res)

    return res


# ==============================================================================
# Check OCR reference charset
def check_alignment_charset(text_to_align: str, dump_charset: bool=False, show_all: bool=False) -> bool:
    """Check the string passed as parameter contains only acceptable
    unicode code points for the alignment.

    The alignment tool (coded before Unicode 2.0 in 1996) currently requires
    that code points are 16 bits wide at most.
    We also add the constraint that we do not want UTF-16 surrogates nor non-characters.

    Args:
        text_to_align (str): text to check before alignment
        dump_charset (bool): whether to dump the charset or not (debug)
        show_all (bool): whether to print info for all chars (not only problematic ones) (debug)

    Returns:
        bool: Return True if the text has a valid charset, False otherwise.
    """
    def _unichr2str(unichar: str) -> str:
        return unicodedata.name(unichar, repr(unichar))

    def _char_str(char_str):
        codepoint = ord(char_str)
        if (char in "\u000A\u000B\u000C\u000D\u0085\u2028\u2029"  # newline
            or 0x0000 <= codepoint <= 0x0009 # C0
            or 0x000B <= codepoint <= 0x001F
            or codepoint == 0x007F
            or 0x0080 <= codepoint <= 0x009F # C1
            ):
            return repr(char_str)
        return char_str

    errors = 0
    for pos, char in enumerate(text_to_align):
        codepoint = ord(char)
        char_str = _char_str(char)
        if (
            codepoint in (0x0000, 0xFFFE, 0xFFFF)  # NULL, non chars
            or 0xD800 <= codepoint <= 0xDFFF  # surrogates
            or codepoint > 0xFFFF):  # large codepoints
            errors += 1
            print(f"Invalid char@{pos:03d}:   ({hex(ord(char))} {_unichr2str(char)}"
                  f" -- cat.: {unicodedata.category(char)})")
        elif not (
            codepoint == ord("\n")
            or 0x0020 <= codepoint <= 0x007E  # Basic_Latin
            or 0x00A0 <= codepoint <= 0x00FF  # Latin-1_Supplement
            or 0x0100 <= codepoint <= 0x017F  # Latin Extended-A
            or 0x0200 <= codepoint <= 0x026F  # General Punctuation
            or codepoint == 0x2014  # — EM DASH
            or codepoint == 0x2029 # Paragraph Separator
            or codepoint in (0x24b6, 0x24b7, 0x24c4, 0x24c5, 0x24cb) # Ⓐ, Ⓑ, Ⓞ, Ⓟ, Ⓥ: Enclosed Alphanumerics (2460–24FF)
            or codepoint in (0x261E, )  # 0x261E:☞ (0x2605: ★): Miscellaneous Symbols (2600–26FF)
            or codepoint in (0x2709, )  # ✉: Dingbats (2700–27BF)
            or 0xE000 <= codepoint <= 0xE0FF  # first 255 chars of Private Use Area (E000–F8FF)
            or 0xFB00 <= codepoint <= 0xFB06  # ﬀ, ﬁ, ﬂ, ﬃ, ﬄ, ﬅ, ﬆ (ligatures)
            or codepoint == 0xFFFD  # � REPLACEMENT CHARACTER
            ):
            print(f"Suspect char@{pos:03d}: {char_str} ({hex(ord(char))} {_unichr2str(char)}"
                  f" -- cat.: {unicodedata.category(char)})")
        elif show_all:
            print(f"  Valid char@{pos:03d}: {char_str} ({hex(ord(char))} {_unichr2str(char)}"
                  f" -- cat.: {unicodedata.category(char)})")
    # if not unicodedata.is_normalized('NFKC', text_to_align):
    #     # Note: circled letter symbols do not seem to comply with NFKC.
    #     #        indeed: unicodedata.normalize('NFKC', "Ⓞ") ==  "O"
    #     # Maybe NFC is enough as we want only composed chars.
    #     print("Warning: string is not normalized (compatibility + composed mode).")
    if not unicodedata.is_normalized('NFC', text_to_align):
        print("Warning: string is not normalized (composed mode).")
    if dump_charset:
        print("Charset:")
        # print("Charset for string:")
        # print(f"\t\"{text_to_align}\"")
        charset = Counter(text_to_align)
        print("    # | Char")
        print("  ----+---------------------")
        for char, count in charset.most_common():
            char_str = _char_str(char)
            try:
                print(f"  {count:3d} | {char_str} ({hex(ord(char))} {_unichr2str(char)}"
                  f" -- cat.: {unicodedata.category(char)})")
            except UnicodeEncodeError:
                print(f"  {count:3d} | ({hex(ord(char))} {_unichr2str(char)}"
                  f" -- cat.: {unicodedata.category(char)})")
        print("")
    return errors == 0


# ==============================================================================
# Fix manual NER XML (raw NER reference)
# Expected input: NER "XML" produced with annotation tool (contains annotation, 
# un-normalized charset but MAY contain XML escaping)
# Output: Clean NER XML to be used as reference (/!\ newlines encoded as PS)
# Transformation: can add, remove and change chars

def fix_manual_ner_xml(ner_xml_orig: str) -> str:
    # Get a clean line
    fixed = remove_unicode_bom(ner_xml_orig)
    fixed = fixed.rstrip()  # removes any training whitespace and newline char

    # Sanity check
    if not check_xml(fixed, taglist=TAG_LIST):
        raise ValueError("Invalid file.")

    # Split on tags
    chunks, tags = chop_on_tags(fixed, tag_list=TAG_LIST)
    assert(len(chunks) == len(tags) + 1)

    # Unescape XML entities if we have some.
    chunks = list(map(xml_unescape, chunks))

    # Replace annotation codes (e.g. "::LH::") with (private if needed) unicode chars
    chunks = list(map(replace_annotation_codes, chunks))
    
    # Simplify unicode charset
    chunks = list(map(simplify_unicode_charset, chunks))

    # Escape XML entities
    # https://www.w3.org/TR/unicode-xml/#Suitable (obsolete?)
    chunks = list(map(xml_escape, chunks))
    
    # Recreate string (recombine chunks and tags)
    final_list = []
    tags.append("")
    for c, t in zip(chunks, tags):
        final_list.append(c)
        final_list.append(t)
    fixed_final = "".join(final_list)

    # Final charset check
    if not check_alignment_charset(fixed_final):
        raise ValueError("Incorrect charset!")

    return fixed_final


def xml_escape(text: str, nl_to_ps: bool=True) -> str:
    # just like https://github.com/python/cpython/tree/3.10/Lib/xml/sax/saxutils.py
    # with extra chars
    esc = text
    esc = esc.replace("&", "&amp;")  # Must be first
    esc = esc.replace("<", "&lt;")
    esc = esc.replace(">", "&gt;")
    esc = esc.replace("\"", "&quot;")
    esc = esc.replace("'", "&apos;")
    if nl_to_ps:
        esc = esc.replace("\n", "\u2029")  # use &#20209; instead?
    return esc


def xml_unescape(xml: str, ps_to_nl: bool=True) -> str:
    # just like https://github.com/python/cpython/tree/3.10/Lib/xml/sax/saxutils.py
    # with extra chars
    unesc = xml
    if ps_to_nl:
        unesc = unesc.replace("\u2029", "\n")  # use &#20209; instead?
    unesc = unesc.replace("&apos;", "'")
    unesc = unesc.replace("&quot;", "\"")
    unesc = unesc.replace("&gt;", ">")
    unesc = unesc.replace("&lt;", "<")
    unesc = unesc.replace("&amp;", "&")  # Must be last
    return unesc

# ==============================================================================
# align NER tags
# Expected input 1: clean final NER XML ref (no annotation codes, escaped entities)
# Expected input 2: final OCR text (no annotation codes, simplified charset)

ALIGNMENT_SIMPLIFICATION_TABLE = str.maketrans({
    "\u00B9": "1", # ¹ U+00B9 to U+0031  1
    "\u00B2": "2", # ² U+00B2 to U+0032  2
    "\u00B3": "3", # ³ U+00B3 to U+0033  3
    "°": "o", # º U+00BA to U+006F  o
    "À": "A",
    "Ç": "C",
    "É": "E",
    "È": "E",
    "Ê": "E",
    "Ë": "E",
    "Î": "I",
    "Ï": "I",
    "Ô": "O",
    "Ö": "O",
    "Û": "U",
    "Ü": "U",
    "à": "a",
    "â": "a",
    "ç": "c",
    "é": "e",
    "è": "e",
    "ê": "e",
    "ë": "e",
    "î": "i",
    "ï": "i",
    "ô": "o",
    "ö": "o",
    "û": "u",
    "ü": "u",
    "@": "a",
    "{": "(",
    "[": "(",
    "}": ")",
    "]": ")",
    ",": ".",
    ":": ".",
    ";": ".",
    "?": "!",
    "\n": " ",  # easier debugging
    "\u2014": "\u002D", # — EM DASH  # /!\ kept because we have special chars like this
    "~": "-",
    "\u0192": "f",  # ƒ (0x192 LATIN SMALL LETTER F WITH HOOK -- cat.: Ll)
    "\u017f": "f",  # ſ 0x17f LATIN SMALL LETTER LONG S
    # Cyrillic (because Pero does output some of these)
    "\u0410": "A",  # 0410 А CYRILLIC CAPITAL LETTER A
    "\u0411": "B",  # 0411 Б CYRILLIC CAPITAL LETTER BE
    "\u0412": "B",  # 0412 В CYRILLIC CAPITAL LETTER VE
    "\u0413": "T",  # 0413 Г CYRILLIC CAPITAL LETTER GHE
    "\u0414": "A",  # 0414 Д CYRILLIC CAPITAL LETTER DE
    "\u0415": "E",  # 0415 Е CYRILLIC CAPITAL LETTER IE
    "\u0416": "X",  # 0416 Ж CYRILLIC CAPITAL LETTER ZHE
    "\u0417": "3",  # 0417 З CYRILLIC CAPITAL LETTER ZE
    "\u0418": "N",  # 0418 И CYRILLIC CAPITAL LETTER I
    "\u0419": "N",  # 0419 Й CYRILLIC CAPITAL LETTER SHORT I
    "\u041A": "K",  # 041A К CYRILLIC CAPITAL LETTER KA
    "\u041B": "A",  # 041B Л CYRILLIC CAPITAL LETTER EL
    "\u041C": "M",  # 041C М CYRILLIC CAPITAL LETTER EM
    "\u041D": "H",  # 041D Н CYRILLIC CAPITAL LETTER EN
    "\u041E": "O",  # 041E О CYRILLIC CAPITAL LETTER O
    "\u041F": "M",  # 041F П CYRILLIC CAPITAL LETTER PE
    "\u0420": "P",  # 0420 Р CYRILLIC CAPITAL LETTER ER
    "\u0421": "C",  # 0421 С CYRILLIC CAPITAL LETTER ES
    "\u0422": "T",  # 0422 Т CYRILLIC CAPITAL LETTER TE
    "\u0423": "y",  # 0423 У CYRILLIC CAPITAL LETTER U
    "\u0424": "O",  # 0424 Ф CYRILLIC CAPITAL LETTER EF
    "\u0425": "X",  # 0425 Х CYRILLIC CAPITAL LETTER HA
    "\u0426": "U",  # 0426 Ц CYRILLIC CAPITAL LETTER TSE
    "\u0427": "y",  # 0427 Ч CYRILLIC CAPITAL LETTER CHE
    "\u0428": "W",  # 0428 Ш CYRILLIC CAPITAL LETTER SHA
    "\u0429": "W",  # 0429 Щ CYRILLIC CAPITAL LETTER SHCHA
    "\u042A": "b",  # 042A Ъ CYRILLIC CAPITAL LETTER HARD SIGN
    "\u042B": "bl",  # 042B Ы CYRILLIC CAPITAL LETTER YERU
    "\u042C": "b",  # 042C Ь CYRILLIC CAPITAL LETTER SOFT SIGN
    "\u042D": "3",  # 042D Э CYRILLIC CAPITAL LETTER E
    "\u042E": "IO",  # 042E Ю CYRILLIC CAPITAL LETTER YU
    "\u042F": "A",  # 042F Я CYRILLIC CAPITAL LETTER YA
    "\u0430": "a",  # 0430 а CYRILLIC SMALL LETTER A
    "\u0431": "b",  # 0431 б CYRILLIC SMALL LETTER BE
    "\u0432": "B",  # 0432 в CYRILLIC SMALL LETTER VE
    "\u0433": "T",  # 0433 г CYRILLIC SMALL LETTER GHE
    "\u0434": "A",  # 0434 д CYRILLIC SMALL LETTER DE
    "\u0435": "e",  # 0435 е CYRILLIC SMALL LETTER IE
    "\u0436": "x",  # 0436 ж CYRILLIC SMALL LETTER ZHE
    "\u0437": "3",  # 0437 з CYRILLIC SMALL LETTER ZE
    "\u0438": "n",  # 0438 и CYRILLIC SMALL LETTER I
    "\u0439": "n",  # 0439 й CYRILLIC SMALL LETTER SHORT I
    "\u043A": "K",  # 043A к CYRILLIC SMALL LETTER KA
    "\u043B": "n",  # 043B л CYRILLIC SMALL LETTER EL
    "\u043C": "m",  # 043C м CYRILLIC SMALL LETTER EM
    "\u043D": "H",  # 043D н CYRILLIC SMALL LETTER EN
    "\u043E": "o",  # 043E о CYRILLIC SMALL LETTER O
    "\u043F": "n",  # 043F п CYRILLIC SMALL LETTER PE
    "\u0440": "p",  # 0440 р CYRILLIC SMALL LETTER ER
    "\u0441": "c",  # 0441 с CYRILLIC SMALL LETTER ES
    "\u0442": "T",  # 0442 т CYRILLIC SMALL LETTER TE
    "\u0443": "y",  # 0443 у CYRILLIC SMALL LETTER U
    "\u0444": "o",  # 0444 ф CYRILLIC SMALL LETTER EF
    "\u0445": "x",  # 0445 х CYRILLIC SMALL LETTER HA
    "\u0446": "u",  # 0446 ц CYRILLIC SMALL LETTER TSE
    "\u0447": "y",  # 0447 ч CYRILLIC SMALL LETTER CHE
    "\u0448": "w",  # 0448 ш CYRILLIC SMALL LETTER SHA
    "\u0449": "w",  # 0449 щ CYRILLIC SMALL LETTER SHCHA
    "\u044A": "b",  # 044A ъ CYRILLIC SMALL LETTER HARD SIGN
    "\u044B": "bl",  # 044B ы CYRILLIC SMALL LETTER YERU
    "\u044C": "b",  # 044C ь CYRILLIC SMALL LETTER SOFT SIGN
    "\u044D": "3",  # 044D э CYRILLIC SMALL LETTER E
    "\u044E": "o",  # 044E ю CYRILLIC SMALL LETTER YU
    "\u044F": "A",  # 044F я CYRILLIC SMALL LETTER YA
    "\u0450": "e",  # 0450 ѐ CYRILLIC SMALL LETTER IE WITH GRAVE
    "\u0451": "e",  # 0451 ё CYRILLIC SMALL LETTER IO
})


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
    chunks = regex.split(r"(</?\L<tag>>)", ner_xml, tag=tag_list) # TODO ignore case for tags?
    A_chunks = chunks[0::2]
    A_tags = [] if len(chunks) < 2 else chunks[1::2]
    return A_chunks, A_tags


def simplify_for_alignment(text: str, case_insensitive=True) -> str:
    # Single char "normalizations"
    # tolerance to OCR single char substitutions (case insensitive, no accents, etc.)
    # which DOES NOT CHANGE THE STRING LENGTH
    simplified = text
    simplified = simplified.translate(ALIGNMENT_SIMPLIFICATION_TABLE)
    if case_insensitive:
        simplified = simplified.upper()
    return simplified


def add_tags_prediction(ner_xml_final: str, text_ocr_final: str, debug=False) -> Tuple[str, bool]:
    """Align NER tag positions from reference to noisy OCR.

    Args:
        ner_xml_final (str): Reference NER XML with correct tags.
        text_ocr_final (str): Noisy OCR to align tags on.
        debug (bool, optional): Activate debug output. Defaults to False.

    Raises:
        ValueError: If XML content is invalid
        ValueError: If NER XML charset is invalid
        RuntimeError: If OCR text normalization changes its length
        ValueError: If OCR text charset is invalid

    Returns:
        Tuple[str, bool]: (xml, no_empty_tag) XML with projected tags and wether this looks correct.
    """
    # 1. Process ner_xml_final
    # => Assume we already have a clean valid XML content with normalized unicode
    # text without any annotation codes
    ## Sanity check
    if not check_xml(ner_xml_final, taglist=TAG_LIST):
        raise ValueError("Invalid XML content.")
    ## Chunking
    A_chunks, A_tags = chop_on_tags(ner_xml_final, tag_list=TAG_LIST)
    ## Replace XML entities
    A_chunks = list(map(xml_unescape, A_chunks))
    ## Simplify the string to ease alignment (e.g. "é"->"e")
    A_chunks = list(map(simplify_for_alignment, A_chunks))
    ## Build string ready for alignment
    a = "".join(A_chunks)
    if not check_alignment_charset(a): # dump_charset=debug):
        raise ValueError("Invalid charset for A, cannot align.")
    # => a is now ready for alignment

    # 2. Process text_ocr_final
    # => Assume we already have a normalized unicode text without any XML content (tags or entities)
    # Simplify the string to ease alignment (e.g. "é"->"e")
    b = text_ocr_final  # We keep b and align on a simplified version of it
    bs = simplify_for_alignment(text_ocr_final)
    if len(b) != len(bs):
        raise RuntimeError("simplify_for_alignment changed string len. Cannot align simplified string.")
    if not check_alignment_charset(bs): # dump_charset=debug):
        raise ValueError("Invalid charset for BS, cannot align.")

    # 3. Align
    if debug:
        A, B = isri_tools.align(a, bs, '␣')
        print("A", A)
        print("B", B)
    A, B = isri_tools.get_align_map(a, bs)
    if debug:
        print("A", A)
        print("B", B)

    # 4. 
    pos_tags = np.cumsum([len(x) for x in A_chunks[:-1]])
    if debug:
        print("pos_tags", pos_tags)

    # 5. Reprojet b on the alignment string
    n = max(np.max(A), np.max(B)) + 1
    chr_list = [ '' for i in range(n + 1)]
    for k, c in zip(B, b):
        chr_list[k] = c
    # print("chr_list", chr_list)
    # print("n", n, "len(chr_list)", len(chr_list))

    # 6. Add tags on the alignment string while escaping
    stack = []
    left = 0
    right = left
    empty_tag = False
    for p, tag in zip(list(pos_tags), list(A_tags)):
        if tag.startswith("</"):
            right = A[p-1] + 1
            if left+1 == right:
                print(f"Error: empty tag {tag} after alignment.")
                empty_tag = True
        else:
            right = A[p]
        sub = chr_list[left:right]
        if debug:
            print(left, right, sub)
        stack.append(xml_escape("".join(sub)))
        stack.append(tag)
        left = right

    right = None
    sub = chr_list[left:right]
    if debug:
        print(left, right, sub)
        print("")
    stack.append(xml_escape("".join(sub)))

    return "".join(stack), not empty_tag








# ==============================================================================
# TODO OCR eval

# TODO OCR equivalences
# TODO case insensitive or not
# TODO check charset
# TODO call isri bindings



OCR_EQUIVALENCE_MAP = [
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
    # ("Æ", "AE"), # U+00C6
    # ("æ", "ae"), # U+00E6
    # ("Œ", "OE"), # U+0152
    # ("œ", "oe"), # U+0153

    ("\u2014", "\u002D"), # — EM DASH  # /!\ kept because we have special chars like this

    # TODO equivalence for cyrillic chars? maybe not, this is due to bad word script decision

    # TODO remove accents? -> extra map
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
    # ("@", "a"),
]





# ==============================================================================
# Read UTF-8 file content (opt. strip trailing \n, delete leading BOM…)
def read_utf8_file(filename: str, strip_training_spaces: bool=False, delete_leading_bom: bool=False) -> str:
    text = None
    with io.open(filename, "rt", encoding="UTF-8", newline=None,
                 errors="strict") as file_input:
        text = file_input.read()
    if delete_leading_bom:
        text = remove_unicode_bom(text)
    if strip_training_spaces:
        text = text.rstrip() # this removes any newline and space char
    return text

def remove_unicode_bom(text: str) -> str:
    if text.startswith("\uFEFF"): # ZERO WIDTH NO-BREAK SPACE // Byte Order Mark
        return text[1:]
    return text




# 8<----------------------------------

# def remove_xml_tags_and_entities(xml: str) -> str:
#     # We reuse existing functions here
#     chunks, _tags = chop_on_tags(xml, tag_list=TAG_LIST)
#     chunks = list(map(xml_unescape, chunks))
#     return "".join(chunks)
