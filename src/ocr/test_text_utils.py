
from text_utils import (
    UNICODE_SIMPLIFICATIONS_SINGLE_CHAR,
    charset_stats,
    ocr_simplifications_for_evaluation, 
    simplify_unicode_charset,
    check_alignment_charset,
    add_tags_prediction,
    xml_contains_empty_tags,
    xml_remove_empty_tags,
    fix_manual_ner_xml)

test_str = ("\U0001F449,AB\u000D\u000Aab "
     + "".join(chr(x) for x in UNICODE_SIMPLIFICATIONS_SINGLE_CHAR.keys()) 
     + "\u0001\u0000-\u0009\u000B -\u001F\u007F \u0080-\u009F \uFE00-\uFE0F "
     + " \uD800-\uDFFF \uFDD0-\uFDEF\uFFFE\uFFFF 123")
    
def test_check_alignment_charset():
    assert(not check_alignment_charset(test_str))

def test_simplify_unicode_charset():
    assert(check_alignment_charset(simplify_unicode_charset(test_str)))

def test_str_rstrip_removes_all_spaces():
    s = "abc\u0009 \u00A0 \u2000 \u2001 \u2002 \u2003 \u2004 \u2005 \u2006 \u2007 \u2008 \u2009 \u200A \u202F \u205F"
    assert(s.rstrip() == "abc")

def test_str_rstrip_removes_all_newlines():
    s = "abc \u000A \u000B \u000C \u000D \u0085 \u2028 \u2029"
    assert(s.rstrip() == "abc")

def test_add_tags_prediction():
    A = "<PER>Anthony</PER> \uE001, <FT>fab.</FT> du <ACT>pêche</ACT>, <LOC>Châtelet</LOC>xx"
    B = "\u261E Mme Antoine, \uE00A fab la p&cheur et \ndu ch>telet et du Faub. St Antoine"
    R = add_tags_prediction(A, B)
    assert(R == "\u261E Mme <PER>Antoine</PER>, \uE00A <FT>fab</FT> la <ACT>p&amp;che</ACT>ur et \u2029du <LOC>ch&gt;telet</LOC> et du Faub. St Antoine")

    A = "<PER>Anthony</PER> \uE001, <FT>fab</FT><FT>.</FT> du <ACT>pêche</ACT>, <LOC>Châtelet</LOC>xx"
    B = "\u261E Mme Antoine, \uE00A fab la p&cheur et \ndu ch>telet et du Faub. St Antoine"
    R = add_tags_prediction(A, B)
    assert(R == "\u261E Mme <PER>Antoine</PER>, \uE00A <FT>fab</FT><FT></FT> la <ACT>p&amp;che</ACT>ur et \u2029du <LOC>ch&gt;telet</LOC> et du Faub. St Antoine")

    A = "<PER>Anthony</PER> \uE001, <FT>fab</FT><FT>.</FT> du <ACT>pêche</ACT>, <LOC>Châtelet</LOC>xx"
    B = ""
    R = add_tags_prediction(A, B)
    assert(R == "<PER></PER><FT></FT><FT></FT><ACT></ACT><LOC></LOC>")



def test_xml_contains_empty_tags():
    XML = "<PER>Lebas (en gros)</PER>,<PER> </PER><LOC>R. et Div. de la Fraternité</LOC>, <CARDINAL>91</CARDINAL>."
    assert(xml_contains_empty_tags(XML))

    XML = "<PER>Lebas (en gros)</PER><PER> Bob</PER>,<LOC>R. et Div. de la Fraternité</LOC>, <CARDINAL>91</CARDINAL>."
    assert(not xml_contains_empty_tags(XML))

def test_xml_remove_empty_tags():
    XML = "<PER>Lebas (en gros)</PER>,<PER>\u2029 </PER><LOC>R. et Div. de la Fraternité</LOC>, <CARDINAL>91</CARDINAL>."
    XML2 = xml_remove_empty_tags(XML)
    XML_EXP = "<PER>Lebas (en gros)</PER>,\u2029 <LOC>R. et Div. de la Fraternité</LOC>, <CARDINAL>91</CARDINAL>."
    assert(XML2 == XML_EXP)

    XML = "<PER>Lebas (en gros)</PER><PER> Bob</PER>,<LOC>R. et Div. de la Fraternité</LOC>, <CARDINAL>91</CARDINAL>."
    XML2 = xml_remove_empty_tags(XML)
    assert(XML2 == XML)

def test_fix_manual_ner_xml():
    # std case with tag removal
    XML = "\n  <PER>Lebas ::MO:: (en gros)</PER><PER>\u2029 </PER><LOC>R. et \nDiv. de la Fraternité</LOC>, <CARDINAL>91</CARDINAL>.\n\n"
    XML2 = fix_manual_ner_xml(XML)
    XML_EXP = "<PER>Lebas \u24c4 (en gros)\u2029 </PER><LOC>R. et \u2029Div. de la Fraternité</LOC>, <CARDINAL>91</CARDINAL>."
    assert(XML2 == XML_EXP)

    # empty case
    XML = ""
    XML2 = fix_manual_ner_xml(XML)
    assert(XML2 == XML)

    # idempotence
    XML = "<PER>Lebas \u24c4 (en gros)</PER>\u2029 <LOC>R. et \u2029Div. de la Fraternité</LOC>, <CARDINAL>91</CARDINAL>."
    XML2 = fix_manual_ner_xml(XML)
    assert(XML2 == XML)

    # corner case: str starts with whitechar: don't remove tag
    XML = "<PER>Dufey fils</PER>, <ACT>bijoutier</ACT>, <LOC>passage de la Réunion</LOC>,<CARDINAL>\u20297</CARDINAL>. 284"
    XML2 = fix_manual_ner_xml(XML)
    assert(XML2 == XML)

    # touching similar tags
    XML = "<PER>Dufey fils</PER><PER>,</PER>ZZ"
    XML2 = fix_manual_ner_xml(XML)
    XML_EXP = "<PER>Dufey fils,</PER>ZZ"
    assert(XML2 == XML_EXP)

    # warning order is important for touching similar tags
    XML = "x<PER> </PER><PER>,</PER> <ACT>bijoutier</ACT>, <LOC>passage de la Réunion</LOC>,<CARDINAL>\u20297</CARDINAL>. 284"
    XML2 = fix_manual_ner_xml(XML)
    XML_EXP = "x <PER>,</PER> <ACT>bijoutier</ACT>, <LOC>passage de la Réunion</LOC>,<CARDINAL>\u20297</CARDINAL>. 284"
    assert(XML2 == XML_EXP)

    # don't remove tags not touching
    XML = "<PER>Dufay</PER>, <ACT>papetier</ACT>, <LOC>r. S.-Martin</LOC>, <CARDINAL>20</CARDINAL>. <CARDINAL>437</CARDINAL>"
    XML2 = fix_manual_ner_xml(XML)
    assert(XML2 == XML)


def test_charset_stats():
    import pandas as pd
    in1 = ["abc"]
    res1 = charset_stats(in1)
    assert(isinstance(res1, pd.DataFrame))
    assert(set(res1["ord"]) == set(ord(x) for x in in1[0]))

    # check invalid chars
    from itertools import chain
    in2 = ["abc", "def\n\uffff\ud800"]
    res2 = charset_stats(in2)
    assert(isinstance(res2, pd.DataFrame))
    assert(set(res2["ord"]) == set(ord(x) for x in chain(*in2)))

    # check generator input
    res3 = charset_stats(s.upper() for s in in2)
    assert(isinstance(res3, pd.DataFrame))
    assert(set(res3["ord"]) == set(ord(x.upper()) for x in chain(*in2)))

def test_ocr_simplifications_for_evaluation():
    text = "Pinchon («J.A.”) [\u00e9l. de Vincent \u043Bet\u2029 \ufb03Augustin}, rue CaumÆœrtin,\nno. —744."
    expected = "pinchon (\"j.a.\") (el. de vincent \u043Bet  ffiaugustin). rue caumaeoertin. no. -744."
    output = ocr_simplifications_for_evaluation(
                text,
                normalize_spaces=True,
                normalize_brackets=True,
                normalize_dashes=True,
                normalize_punctuation=True,
                normalize_quotes=True,
                casefold=True,
                remove_accents=True,
                skip_charset_norm=False)
    assert(output == expected)

    # idempotence
    expected = "pinchon (\"j.a.\") (el. de vincent \u043Bet  ffiaugustin). rue caumaeoertin. no. -744."
    output = ocr_simplifications_for_evaluation(
                expected,
                normalize_spaces=True,
                normalize_brackets=True,
                normalize_dashes=True,
                normalize_punctuation=True,
                normalize_quotes=True,
                casefold=True,
                remove_accents=True,
                skip_charset_norm=False)
    assert(output == expected)

    # special casefold case: casefold changes string length
    text = "aA\u0130bBéÉ"  # U+0130 LATIN CAPITAL LETTER I WITH DOT ABOVE
    output = ocr_simplifications_for_evaluation( text, casefold=True, remove_accents=False)
    expected = "aa\u0069\u0307bbéé"
    assert(output == expected)

    text = "aA\u0130bBéÉ"  # U+0130 LATIN CAPITAL LETTER I WITH DOT ABOVE
    output = ocr_simplifications_for_evaluation( text, casefold=False, remove_accents=True)
    expected = "aAIbBeE"
    assert(output == expected)

    text = "aA\u0130bBéÉ"  # U+0130 LATIN CAPITAL LETTER I WITH DOT ABOVE
    output = ocr_simplifications_for_evaluation( text, casefold=True, remove_accents=True)
    expected = "aaibbee"
    assert(output == expected)

    # More space normalization
    text = " aaa  aa   ( aaaaa a 2 )\u2029\n   aaa aa.\n  * "
    output = ocr_simplifications_for_evaluation(text, 
                normalize_spaces=False,
                deduplicate_spaces=False,
                apply_french_spacing_rules=False,
                skip_charset_norm=True)
    # no change requested but newline normalization is not optional yet
    expected = " aaa  aa   ( aaaaa a 2 )\n\n   aaa aa.\n  * "
    assert(output == expected)

    output = ocr_simplifications_for_evaluation(text, 
                normalize_spaces=False,
                deduplicate_spaces=True,
                apply_french_spacing_rules=False,
                skip_charset_norm=True)
    expected = "aaa aa ( aaaaa a 2 )\naaa aa.\n*"
    assert(output == expected)

    output = ocr_simplifications_for_evaluation(text, 
                normalize_spaces=False,
                deduplicate_spaces=False,
                apply_french_spacing_rules=True,
                skip_charset_norm=True)
    expected = " aaa  aa (aaaaa a 2) aaa aa.\n  * "
    assert(output == expected)

    output = ocr_simplifications_for_evaluation(text, 
                normalize_spaces=False,
                deduplicate_spaces=True,
                apply_french_spacing_rules=True,
                skip_charset_norm=True)
    expected = "aaa aa (aaaaa a 2) aaa aa.\n*"
    assert(output == expected)

    output = ocr_simplifications_for_evaluation(text, 
                normalize_spaces=True,
                deduplicate_spaces=True,
                apply_french_spacing_rules=True,
                skip_charset_norm=True)
    expected = "aaa aa (aaaaa a 2) aaa aa. *"
    assert(output == expected)


