#!/bin/sh

dpkg-buildpackage -I.git -rfakeroot -us -uc
