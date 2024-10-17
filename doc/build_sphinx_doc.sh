#!/bin/sh
set -e

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
cd doc/sphinxdoc
echo "******** GENERATE ALGORITHMS REFERENCE ********"
# force using default python3 if the the argument is not supplied
if [ -z "$1" ]
    then
        python3 generate_reference.py
    else
        "$1" generate_reference.py
fi

echo "******** BUILDING SPHINX DOCUMENTATION ********"
pandoc ../../FAQ.md -o FAQ.rst
pandoc research_papers.md -o research_papers.rst
jupyter nbconvert ../../src/examples/python/*.ipynb --to rst --output-dir .

make clean
make html

# remove generated algorithm reference rst and temporary html files
rm -r reference
rm -r _templates/reference
rm -r _templates/algo_description_layout_std.html
rm -r _templates/algo_description_layout_streaming.html
rm -r _templates/algorithms_reference.html

# remove rst files generated from markdown
rm FAQ.rst
rm research_papers.rst
rm -r tutorial_*.rst tutorial_*_files

# and copy doxygen into sphinx resulting dir
echo "******** MERGING DOCUMENTATION ********"
rm -fr _build/html/doxygen
cp -R ../../build/doc/doxygen/html _build/html/doxygen
