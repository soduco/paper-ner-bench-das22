#!/bin/bash

# Let's get paranoid
set -eu

######################

WORKDIR=data/gold

mkdir -p $WORKDIR

pushd $WORKDIR > /dev/null

echo "Creating the gold dataset"

# Gather sampled pages numbers & directory name
declare -A directories
directories=()
while IFS= read -r row;
do
  arr=($row)
  directories+=([${arr[0]}]=${arr[1]})
done < sampled_pages.txt

# Creates the gold dataset from a set of directories jsons and stores it in CSV where:
# column 1 is a pre-annotated directory entry in XML
# column 2 is the entry's directory name
rm -f gold.csv
for f in $(ls *.json);
do
  pagenum=$(echo ${f%.*} | sed 's/^0*//')
  directory="${directories[$pagenum]}"
  cat $f | jq --raw-output --arg directory "$directory" '.[] | select(.type=="ENTRY" and .checked==true) | [.ner_xml,$directory] | @csv' >> gold.csv
done

popd  > /dev/null
