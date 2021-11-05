#!/bin/bash

set -e

auth_token=$1
base_url="https://apps.lrde.epita.fr:443/soduco/directory-annotator/directories"

download_img() {
    curl -s -H "Authorization: $auth_token" ${view_url}/image > ${view_name}.jpg
}

download_json(){
    curl -s -H "Authorization: $auth_token" ${view_url}/annotation > ${view_name}.json
}

sample() {
    [ -d samples ] && (rm samples/*) || mkdir samples
    cp /dev/null samples/views.txt
    while IFS=, read -r dir view 
    do
        echo -n "$dir, $view"
        echo -e
        for i in $(seq 1 $samples); do            
            view_url="${base_url}/${dir}.pdf/${view}"
            view_name="samples/${dir}_${view}"
            download_img && \
            download_json && \
            echo "$dir, $view" >> samples/downloaded.txt
        done
    done < samples.csv
}


echo 'Usage: download.sh AUTH_TOKEN'

sample "${@}"

pushd ./samples
convert $(ls -1 | egrep '\.jpg' | tr '\n' ' ') merged.pdf
popd
