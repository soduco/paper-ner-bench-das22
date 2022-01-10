#!/usr/bin/env bash

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This script creates the gold dataset CSV from a set of JSON pages
# 
# Requires env. variable $DATADIR.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ==============================================================================
# Let's get paranoid
set -eu

# Constants
WORKDIR=$DATADIR/gold

# Globals
declare -A directories
# ==============================================================================

# ------------------------------------------------------------------------------
function read_groups() {
  # Gather sampled pages numbers & directory name
  directories=()
  while IFS= read -r row;
  do
    arr=($row)
    directories+=([${arr[0]}]=${arr[1]})
  done < sampled_pages.txt
}


# ------------------------------------------------------------------------------
function create_gold() {
  # Creates the gold dataset from a set of directories jsons and stores it 
  # as a CSV where:
  # - column 1 is a pre-annotated directory entry in XML
  # - column 2 is the entry's directory name
  rm -f gold.csv
  for f in $(ls *.json);
  do
    pagenum=$(echo ${f%.*} | sed 's/^0*//')
    directory="${directories[$pagenum]}"
    cat $f | jq --raw-output --arg directory "$directory" '.[] | select(.type=="ENTRY" and .checked==true) | [.ner_xml,$directory] | @csv' >> gold.csv
  done
}


# ==============================================================================
# Entry point
mkdir -p $WORKDIR
pushd $WORKDIR > /dev/null
read_groups
create_gold
popd  > /dev/null

