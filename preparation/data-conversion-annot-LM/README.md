# Annotation convertion

To convert the old ASCII tag format into the new XML tag format.

## Main script
The script is `convert_annot_lm.py` and is self documented (online help).

Synopsis:
```
$ python convert_annot_lm.py --help
usage: convert_annot_lm.py [-h] input_file output_file

Convert ASCII tag format to XML tag format.

positional arguments:
  input_file   Path to input file.
  output_file  Path to output file.

optional arguments:
  -h, --help   show this help message and exit
```

## Content conversion
It takes a json export of an annotation file which contains content of the following kind:
```json
"text_ocr": "##Castries (Mis de) C.## µµ::LH::µµ, ££mar. de camp££, @@Varennes@@, $$22$$.\n",
```

and converts it to:
```json
"text_ocr": "Castries (Mis de) C. ::LH::, mar. de camp, Varennes, 22.\n",
"ner_xml": "<PER>Castries (Mis de) C.</PER> <TITRE>::LH::</TITRE>, <ACT>mar. de camp</ACT>, <LOC>Varennes</LOC>, <CARD>22</CARD>.\n",
```

## Batch download and convert files
```shell
export auth_token=YOURTOKEN
export base_url="https://apps.lrde.epita.fr/soduco/directory-annotator/storage/directories"
export dest_dir="output"
# filelist.csv contains lines like:
# Bottin1_1837,80
while IFS=, read -r dir view 
do
    view_url="${base_url}/${dir}.pdf/${view}"
    file_basename="${dir}_${view}"
    echo -n "$dir, $view -> file_name"
    echo -e
    curl -s -H "Authorization: $auth_token" ${view_url}/annotation | jq '.["content"]' > "${dest_dir}/${file_basename}.json" && \
    python convert_annot_lm.py "${dest_dir}/${file_basename}.json" "${dest_dir}/${file_basename}-converted.json"
done < filelist.csv
```
