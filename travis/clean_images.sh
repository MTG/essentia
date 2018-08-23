#!/usr/bin/env bash
set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"                                                           

"$DIR/clean_containers.sh"
echo $(docker images | egrep "hello" | awk '{print $3}') | while read image ; do
    echo $image
    docker rmi $image
done
