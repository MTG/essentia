#!/usr/bin/env bash
set -euo pipefail

echo $(docker ps -a | tail -n +2 | egrep "hello" | awk {'print $1'}) | while read container_id; do
    if [ ! -z "$container_id" ] ; then
        docker rm "$container_id" | sed 's/^/    /'
    fi
done
