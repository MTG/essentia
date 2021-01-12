#!/usr/bin/env bash
set -e
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

echo "Building gaia $GAIA_VERSION"

curl -SLO https://github.com/MTG/gaia/archive/v$GAIA_VERSION.tar.gz
tar -xf v$GAIA_VERSION.tar.gz
cd gaia-*/

python3 ./waf configure --prefix=$PREFIX
python3 ./waf
python3 ./waf install

cd ../..
rm -rf tmp
