
A set of raw directories entries texts extracted with Pero-OCR, disjoint from the gold dataset and selected to pre-train BERT.

The file `pretrain.txt` contains all non-empty entries that are at least 10 chars long.
Line breaks within entries are replaced with whitechars so one line corresponds to one entry.

To re-create `pretrain.txt` from jsons: 
```bash
OUT=pretrain.txt rm -f $OUT ;  jq --raw-output '.[] | select(.type=="ENTRY") | .text_ocr | gsub("\n";" ")  | select(length > 10 )' *.json > $OUT && sed -i '/^$/d' $OUT
```
