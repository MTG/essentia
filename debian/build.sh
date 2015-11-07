cd ..
./waf configure --mode=release --with-python --with-vamp --prefix=./debian/essentia/usr/
./waf
./waf install
cd -

mv essentia/usr/lib/python2.7/site-packages/ essentia/usr/lib/python2.7/dist-packages
dpkg-deb --build essentia
