# OCR evaluation and text synchronization

Reference issue: https://github.com/soduco/paper-ner-bench-das22/issues/3

Tasks:
- [ ] code for OCR evaluation
- [ ] code for text normalization
- [ ] code for text sanity checks
- [ ] code for projecting tag positions in NER_XML_REF to NER_XML_PRED

# Usage

## Building

```sh
python setup.py build_ext --inplace
```

## Usage (Python API)

See the demo [notebook](./DEMO.ipynb).

## Usage (Command Line)

```sh
usage: ner_sync_ref.py [-h] [-d] reference.xml predicted.txt
```


* **reference**    Input reference file with UTF-8 encoding and XML tags.
* **predicted**    Input prediction file with UTF-8 encoding (and optionally XML tags).
