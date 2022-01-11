
from text_utils import (
    UNICODE_SIMPLIFICATIONS_SINGLE_CHAR, 
    simplify_unicode_charset,
    check_alignment_charset,
    add_tags_prediction)

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

def test_align_ner_1():
    A = "<PER>Anthony</PER> \uE001, <FT>fab.</FT> du <ACT>pêche</ACT>, <LOC>Châtelet</LOC>xx"
    B = "\u261E Mme Antoine, \uE00A fab la p&cheur et \ndu ch>telet et du Faub. St Antoine"
    R, no_empty_tag = add_tags_prediction(A, B)
    assert(no_empty_tag)
    assert(R == "\u261E Mme <PER>Antoine</PER>, \uE00A <FT>fab</FT> la <ACT>p&amp;che</ACT>ur et \u2029du <LOC>ch&gt;telet</LOC> et du Faub. St Antoine")

def test_align_ner_1():
    A = "<PER>Anthony</PER> \uE001, <FT>fab</FT><FT>.</FT> du <ACT>pêche</ACT>, <LOC>Châtelet</LOC>xx"
    B = "\u261E Mme Antoine, \uE00A fab la p&cheur et \ndu ch>telet et du Faub. St Antoine"
    R, no_empty_tag = add_tags_prediction(A, B)
    assert(not no_empty_tag)
    assert(R == "\u261E Mme <PER>Antoine</PER>, \uE00A <FT>fab</FT><FT></FT> la <ACT>p&amp;che</ACT>ur et \u2029du <LOC>ch&gt;telet</LOC> et du Faub. St Antoine")
