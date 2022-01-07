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
import logging
import argparse
import sys
import unicodedata

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
# FIXME add cyrillic as Pero can output content in this script (actually add entire scripts)
#       OR project extra char to unused char in GT (Unicode PUA?)
ALLOWED_INPUT = (
    """ !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abc"""
    """defghijklmnopqrstuvwxyz{|}~ ¡¢£¤¥¦§¨©ª«¬­®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉ"""
    """ÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿŒœŠšŸŽžƒˆ˜–—‘’"""
    """‚“”„†‡•…‰‹›€™ﬁﬂﬀﬃﬄ"""
# Horizontal tab added separately for convenience
    "\u0009"
# additions due to Unicode normalization:
# - 0308 COMBINING DIAERESIS
# - 0301 COMBINING ACUTE ACCENT
# - 03BC GREEK SMALL LETTER MU
# - 0327 COMBINING CEDILLA
# - 0303 COMBINING TILDE
# - 0304 COMBINING MACRON
# - 2044 FRACTION SLASH
    "\u0308\u0301\u03BC\u0327\u0303\u0304\u2044"
# ZERO WIDTH NO-BREAK SPACE
    "\ufeff"
    )

TRANSFORMATIONS = [
    ("\u0009", " "), # HTAB to SPACE
    ("\u00A0", " "), # NBSP to SPACE
    ("¦", "|"), # U+00A6 # Commonly interchanged
# ¨ U+00A8 to U+0020 U+0308    ̈
# ª U+00AA to U+0061  a
    ("«", "\""), # U+00AB
    ("\u00AD", ""), # remove SOFT HYPHEN
# ¯ U+00AF to U+0020 U+0304    ̄
# ² U+00B2 to U+0032  2
# ³ U+00B3 to U+0033  3
# ´ U+00B4 to U+0020 U+0301    ́
# µ U+00B5 to U+03BC  μ
# ¸ U+00B8 to U+0020 U+0327    ̧
# ¹ U+00B9 to U+0031  1
# º U+00BA to U+006F  o
    ("»", "\""), # U+00BB
# ¼ U+00BC to U+0031 U+002F U+0034  1/4 # done by FRACTION SLASH replace after norm.
# ½ U+00BD to U+0031 U+002F U+0032  1/2 # done by FRACTION SLASH replace after norm.
# ¾ U+00BE to U+0033 U+002F U+0034  3/4 # done by FRACTION SLASH replace after norm.
    ("Æ", "AE"), # U+00C6
    ("æ", "ae"), # U+00E6
    ("Œ", "OE"), # U+0152
    ("œ", "oe"), # U+0153
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
    ("\ufeff", ""), # ZERO WIDTH NO-BREAK SPACE // Byte Order Mark
    ]

CHAR_ERR_LIM = 5

# ==============================================================================
def _transform(unistr):
    s2 = unistr
    for fr, to in TRANSFORMATIONS:
        s2 = s2.replace(fr, to)
    return s2

# ==============================================================================
def project_charset():
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



def ner_sync_ref(reference_with_tags: str, predicted: str) -> str:
    # TODO stub
    raise NotImplementedError()
    # with ref:
    # - unescape XML entities (if any) BUT NOT THE ONES CLASHING WITH XML TAGS
    # - perform checks (charset validity, opt. dump charset)
    # - perform unicode normalization
    # - find and store tag positions, and remove tags
    # - unescape last XML entities ("<", ">" and "&")
    # - recheck (charset validity, opt. dump charset)
    # with predicted:
    # - perform checks (charset validity, opt. dump charset)
    # - perform unicode normalization
    # - recheck (charset validity, opt. dump charset)
    # call tag sync function
    # - /!\ must ensure that ord(c) < (2**16 - 1) for all c
    #   * because of CPP code of the wrapper which uses wchar (16 bits on Windows: bad, 32 on other platforms: good)
    #   * because of internal representation of codepoints on 16 bits (will need a full fix to comply with post 1996 Unicode…)
    # reinsert tags from the end of the string, escaping ("<", ">" and "&") in any substring
    # (or just split at the insert indexes, escape, then recombine with tags inserted…)
    # return result (normalized, predicted string with extra tags inside and ("<", ">" and "&") escaped)

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
    dbg_head_pre = " " * (max(0, (_DBGLINELEN - len(dbg_head)))/2)
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

def _unichr2str(unichar):
    return unicodedata.name(unichar, repr(unichar))

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
    parser.add_argument('output',  #  TODO make it optional (write to stdout in this case)
        help="Path to synchronized output file (predicted text with reference tags synchronized).")
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
    charset = ALLOWED_INPUT + u'\u000a' # Tolerate '\n' (LF) EOL

    err_count = 0
    logger.debug("--- Process started. ---")

    # TODO WIP

    # Read raw files (remove tags from predicted file if any and unescapte XML entities if any)
    # ---> print a warning if using XML content to prevent accidents, but ease use
    # Call alignment function ner_sync_ref(ref, pred) -> pred_with_aligned_tags
    # Write result to output

    logger.debug("--- Process complete. ---")
    # --------------------------------------------------------------------------

    ret_code = ERRCODE_OK
    if err_count > 0:
        logger.error(_DBGSEP)
        logger.error("Input file contains %d illegal characters.", err_count)
        ret_code = ERRCODE_EXTRACHAR
    else:
        logger.debug("Input file contains only legal characters.")

    logger.debug("Clean exit.")
    logger.debug(_DBGSEP)
    return ret_code
    # --------------------------------------------------------------------------

# ==============================================================================
# Entry point
if __name__ == "__main__":
    sys.exit(main())
