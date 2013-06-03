#!/bin/sh


# update Essentia version number in the Doxyfile
cp doc/Doxyfile doc/Doxyfile.tmp
cat doc/Doxyfile.tmp | sed "s/^PROJECT_NUMBER .*$/PROJECT_NUMBER  = $(cat VERSION)/" > doc/Doxyfile
rm doc/Doxyfile.tmp

# update Essentia version number in the Sphinx conf file
cp doc/sphinxdoc/conf.py doc/sphinxdoc/conf.py.tmp
cat doc/sphinxdoc/conf.py.tmp | sed "s/^version = .*$/version = '$(cat VERSION)'/" | sed "s/^release = .*$/release = '$(cat VERSION)'/" > doc/sphinxdoc/conf.py
rm doc/sphinxdoc/conf.py.tmp



# first build doxygen reference
echo "******** BUILDING DOXYGEN DOCUMENTATION ********"
mkdir -p build/doc/doxygen

# call Doxygen
doxygen doc/Doxyfile


# now build Sphinx doc
echo "******** BUILDING SPHINX DOCUMENTATION ********"
cd doc/sphinxdoc
python generate_reference.py
make html

# and copy doxygen into sphinx resulting dir
echo "******** MERGING DOCUMENTATION ********"
rm -fr _build/html/doxygen
cp -R ../../build/doc/doxygen/html _build/html/doxygen
