'''
Module for Text Alignment and OCR Metrics
'''



import isri_tools
from typing import List, Union
from dataclasses import dataclass

@dataclass
class AlignmentStats:
    """Provide statistics about an aligment
    """
    errors: int = 0
    matches: int = 0
    reference_length: int = 0
    x_length: int = 0

    @property
    def length(self):
        return self.errors + self.matches

    @property
    def precision(self) -> float:
        """Get the alignment recall (ratio match/reference length)

        Returns:
            float: Precision score
        """
        return self.matches / self.x_length
    
    @property
    def recall(self) -> float:
        '''
        Get the alignment recall (ratio match/reference length)

        Returns:
            float: Recall score
        '''
        return self.matches / self.reference_length

    @property
    def accuracy(self) -> float:
        '''
        Get the alignment accuracy (number of match over the alignment length)
        '''
        return self.matches / self.length

    
    @property
    def CER(self) -> float:
        '''
        Get the alignment error rare (number of errors over the reference length)
        https://towardsdatascience.com/evaluating-ocr-output-quality-with-character-error-rate-cer-and-word-error-rate-wer-853175297510#5aec
        '''
        return self.errors / self.reference_length

    @property
    def CERnorm(self) -> float:
        '''
        Get the alignment error rare (number of errors over the reference length)
        https://towardsdatascience.com/evaluating-ocr-output-quality-with-character-error-rate-cer-and-word-error-rate-wer-853175297510#5aec
        https://sites.google.com/site/textdigitisation/qualitymeasures/computingerrorrates
        '''
        return self.errors / self.length


@dataclass
class AlignmentReport:
    '''
    Provide an alignment report between a reference string and another string
    '''
    reference: str
    x: str
    aligned_ref: str
    aligned_x: str

    def __init__(self, reference: str, x: str) -> None:
        self.reference = reference
        self.x = x
        self.aligned_ref, self.aligned_x = isri_tools.align(reference, x, "⌴")

        match = 0
        for a,b in zip(self.aligned_ref, self.aligned_x):
            match += int(a == b)

        n = len(self.aligned_ref)
        self._stats = AlignmentStats(matches=match,
                                    errors=n-match,
                                    x_length=len(x),
                                    reference_length=len(reference))

    @property
    def stats(self) -> AlignmentStats:
        return self._stats

def align(reference: str, x: str) -> AlignmentReport:
    return AlignmentReport(reference, x)