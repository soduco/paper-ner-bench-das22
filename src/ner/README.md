
# Evaluating NER approaches against historical directories

This repository contains the code needed to replicate experiments 1 & 2.

Three models are evaluated:
- NER pipeline from Spacy (Tok2vec + CNN)
- [CamemBERT fine-tuned for entity recognition on wikiner-fr](https://huggingface.co/Jean-Baptiste/camembert-ner).
- CamemBERT fine-tuned for entity recognition on wikiner-fr + pretrained on 20,000 raw entries extracted in the directories.

**Experiment 1: prediction performance with decreasing training set size**
1. The gold dataset is split into a train set (~80% of all directory entries) and a test set (~20%). Entries are distributed among the two datasets so the same directory will not appear in both.
2. The trainset is iteratively split in half to create 9 subsets of decreasing sizes (100%, 50%, 25%, 12.5%, 6.2%, 3.1%, 1.5%, 0.8%). The smaller set should contain 24 entries, the larger 6373. Resampling is stratified on entries's source directory to ensure that both train & dev reflect the diversity of directories in the original train set.
3. Models are trained on each sub-trainset. Precision, recall, F1 score and accuracy are measured against both the dev & test sets.

**Experiment 2: prediction performance on noisy OCR data vs clean data**
 :construction: :construction:

## Usage

### Preparation
1. Dependencies
```bash
# /!\ ~1.3G
pip install -r requirements.txt
```

You will also need the SpaCy pipeline optimized for the French language:
```bash
# /!\ 546MB
python -m spacy download fr_core_news_lg
```