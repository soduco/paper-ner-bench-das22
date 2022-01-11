
from text_utils import UNICODE_SIMPLIFICATIONS_SINGLE_CHAR, simplify_unicode_charset, check_alignment_charset

test_str = ("\U0001F449,AB\u000D\u000Aab "
     + "".join(chr(x) for x in UNICODE_SIMPLIFICATIONS_SINGLE_CHAR.keys()) 
     + "\u0001\u0000-\u0009\u000B -\u001F\u007F \u0080-\u009F \uFE00-\uFE0F "
     + " \uD800-\uDFFF \uFDD0-\uFDEF\uFFFE\uFFFF 123")
    
def check_alignment_charset_test():
    assert(not check_alignment_charset(test_str))

def simplify_unicode_charset_test():
    assert(check_alignment_charset(simplify_unicode_charset(test_str)))
