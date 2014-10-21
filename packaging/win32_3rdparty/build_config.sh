#!/bin/bash

HOST=i686-w64-mingw32
PREFIX=`pwd`
echo Installing to: $PREFIX

#SHARED_OR_STATIC="
#--enable-shared \
#--disable-static
#"

SHARED_OR_STATIC="
--disable-shared \
--enable-static
"
