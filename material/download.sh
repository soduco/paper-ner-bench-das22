#!/bin/bash

set -e

auth_token=$1
base_url="https://apps.lrde.epita.fr:443/soduco/directory-annotator/directories"

download_img() {
    curl -s -H "Authorization: $auth_token" ${view_url}/image > ${file_name}.jpg
    echo ""
}

download_json(){
    curl -s -H "Authorization: $auth_token" ${view_url}/annotation > ${file_name}.json
}

sample() {
    [ -d samples ] && (rm samples/*) || mkdir samples
    # cp /dev/null samples/views.txt
    index=1
    while IFS=, read -r dir view 
    do
        echo -n "$index <- $dir, $view"
        echo -e
        view_url="${base_url}/${dir}.pdf/${view}"
        # file_name="samples/${dir}_${view}"
        view_name=$(printf "%04d" $index)
        file_name="samples/$view_name"
        download_img && \
        download_json && \
        echo "${view_name},$dir,$view" >> samples/downloaded.txt
        index=$((index+1))
    done < samples.csv
}


echo 'Usage: download.sh AUTH_TOKEN'

sample "${@}"

pushd ./samples
echo "Merging"
# To get convert to work, I had to edit /etc/ImageMagick-6/policy.xml
# and change the following lines
# <policy domain="resource" name="memory" value="2GiB"/> (previously value="256MiB")
# <policy domain="resource" name="disk" value="2GiB"/> (previously value="1GiB")
# <policy domain="coder" rights="read | write" pattern="PDF" /> (previously rights="none")
convert $(cut -d, -f1 downloaded.txt | sed 's/$/.jpg/g') merged.pdf
cut -d, -f1 downloaded.txt  | sed 's/$/.json/g' | zip merged.zip -@
echo "Cleaning"
rm *.json *.jpg
popd
