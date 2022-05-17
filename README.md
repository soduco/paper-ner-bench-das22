# paper-ner-bench-das22
Sources (latex and code?) for our DAS 2022 paper (NER benchmark)

## Easy-to-download items

- Dataset: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6394464.svg)](https://doi.org/10.5281/zenodo.6394464)
- Paper's PDF: HAL | arXiv | GitHub Release
- Supplementary material: GitHub Release




## Code
Code is in `src/` for the most part. I guess some of the code is still on python notebooks ?

Depending on the size of the dataset, we will either share it with GitHub, or with Zenodo.

Pour l'instant l'asset des images + vignettes est ici:
https://cloud.lrde.epita.fr/s/a8CXFxz4954SDMw

## Models
Huggingface models are shared on Huggingface Hub:
- CamemBERT simple: ::TODO:: 
- CamemBERT pretrained: https://huggingface.co/HueyNemud/berties-pretrained-das22
Users can simple import those models in Python using `model.from_pretrained('berties-pretrained-das22')`

The SpaCy best model is not shared. It's 630+Mb large so storing it in this repo is a bad idea.
Where do we share it (if so) ?


## Latex sources

Structure:
- latex sources are in `src-latex`
- main latex file for the paper is `src-latex/main-paper.tex`
- sub-parts are under `src-latex/parts/`
- some supplementary material is available in `src-latex/main-supplementary-material.tex`

## Interesting related work
- http://spacetime.nypl.org/city-directory-meetup
