Frequently Asked Questions
==========================

How to compile my own C++ code that uses Essentia?
--------------------------------------------------


Here is an example how to compile
[standard_mfcc.cpp](https://github.com/MTG/essentia/blob/2.0.1/src/examples/standard_mfcc.cpp) example on Linux linking with a system-wide installation of Essentia (done by ```./waf install```) and all its dependencies. Modify to your needs. 

```
g++ -pipe -Wall -O2 -fPIC -I/usr/local/include/essentia/ -I/usr/local/include/essentia/scheduler/ -I/usr/local/include/essentia/streaming/  -I/usr/local/include/essentia/utils -I/usr/include/taglib -I/usr/local/include/gaia2 -I/usr/include/qt4 -I/usr/include/qt4/QtCore -D__STDC_CONSTANT_MACROS standard_mfcc.cpp -o standard_mfcc -L/usr/local/lib -lessentia -lfftw3 -lyaml -lavcodec -lavformat -lavutil -lsamplerate -ltag -lfftw3f -lQtCore -lgaia2
```

Alternatively, if you want to create and build your own examples, the easiest way is to add them to ```src/examples``` folder and modify ```src/examples/wscript``` file accordingly.


