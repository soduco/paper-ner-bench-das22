# Internal dataset we used for this paper

**Please do not use this dataset except to check our results: use the official, cleaned and well-documented dataset shared on Zenodo instead:**  
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6394464.svg)](https://doi.org/10.5281/zenodo.6394464)


## Copyright and License
The images were extracted from the original source https://gallica.bnf.fr, owned by the *Bibliothèque nationale de France* (French national library).
Original contents from the *Bibliothèque nationale de France* can be reused non-commercially, provided the mention "Source gallica.bnf.fr / Bibliothèque nationale de France" is kept.  
**Researchers do not have to pay any fee for reusing the original contents in research publications or academic works. ** 
*Original copyright mentions extracted from https://gallica.bnf.fr/edit/und/conditions-dutilisation-des-contenus-de-gallica on March 29, 2022.*

The original contents were significantly transformed before being included in this dataset.
All **derived content** is licensed under the permissive **Creative Commons Attribution 4.0 International** license.


## Content of this folder
For completeness, here is a content summary of the current folder:

- `supervised/00-dataset_article_das22.pdf/`:
  Raw JSON objects exported from our storage system after human annotation.
  There are some buggy regions which were later filtered.
- `supervised/00-deskewed-page-images/`:
  Images exactly as they were presented to used when annotated.
  There color and skew differs from original content.
- `supervised/00-ocr-pero-raw-json/`:
  Raw OCR output for each entry from PERO OCR engine.
- `supervised/00-ocr-tess-raw-json/`:
  *Same for Tesseract v4 engine.*
- `supervised/01-ocr-kraken-raw-json/`:
  *Same for Kraken engine.*
- `supervised/10-ref-ocr-ner-json/`:
  Cleaned-up human annotations: for each entry we provide the bounding box, whether the box is valid, the directory, the page, human-labeled text, human-label entities.
- `supervised/21-ocr-pero-final/`:
  Normalized OCR output for PERO OCR engine.
- `supervised/22-ocr-tess-final/`:
  *Same for Tesseract v4 engine .*
- `supervised/23-ocr-krak-final/`:
  *Same for Kraken engine.*
- `supervised/31-ner_align_pero/`:
  Reference entities from human labels projected onto PERO OCR predictions.
- `supervised/32-ner_align_tess/`:
  *Same for Tesseract v4 engine .*
- `supervised/33-ner_align_krak/`:
  *Same for Kraken engine.*
- `supervised/40-ner_aligned_valid_subset/`:
  Selection of NER targets valid for reference, PERO OCR and Tesseract v4.
- `supervised/41-ner_aligned_valid_subset_with_kraken/`:
  Selection of NER targets valid for reference, PERO OCR, Tesseract v4 **and Kraken**.
- `supervised/80-ocr-text-files/`:
  OCR predictions as a separate file for each entry, to test our Python wrapper for UNLV-ISRI OCR evaluation tools.
- `supervised/81-eval-ocr-files/`:
  Evaluation outputs from UNLV-ISRI OCR evaluation tools for PERO and Tesseract systems.
- `supervised/annotation_table.csv`:
  Our internal annotation tracking table, which contains extra comments about each page (even those not included in the final dataset).
- `unsupervised_pretraining/00-raw_json.tar.gz`:
  Raw entries detected from our platform, with PERO OCR predictions, without human correction, for approx. 7000 pages.
- `unsupervised_pretraining/10-normalized/`:
  Normalized and cleaned file from `unsupervised_pretraining/00-raw_json.tar.gz`
  