#!/usr/bin/env python

# Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
#
# This file is part of Essentia
#
# Essentia is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation (FSF), either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the Affero GNU General Public License
# version 3 along with this program. If not, see http://www.gnu.org/licenses/



import os, subprocess, glob, sys, time
from os.path import join

PKG_DIR = '/essentia_pkg'

def copy_examples(essentia_root_dir, build_dir, target_dir):
    # create target directory
    ret = subprocess.call(['mkdir', target_dir])
    # get binaries:
    programs = [file for file in glob.glob(join(build_dir,'*')) if ".o" not in file]
    # get sources:
    examples_dir = join(essentia_root_dir, 'src', 'examples')
    headers = glob.glob(join(examples_dir,'*.h'))
    sources = glob.glob(join(examples_dir,'*.cpp'))
    libvamp = glob.glob(join(examples_dir, 'libvamp_essentia.dylib'))
    python_examples = glob.glob(join(examples_dir, 'python'))
    allfiles = headers + sources + programs + python_examples + libvamp
    ret = 0
    for file in allfiles:
        print 'copying:', file, 'to', target_dir
        ret += subprocess.call(['cp', '-rf', file, target_dir])

    return ret

def copy_documentation(source, target):
    documentation_dir = join(source, 'build', 'doc', 'doxygen')
    target = join(target, 'documentation')
    return subprocess.call(['cp', '-r', documentation_dir, target])

def copy_python_tests(source, target):
    test_dir = join(source, 'test')
    ret = subprocess.call(['cp', '-r', test_dir, target])
    ret += subprocess.call(['mv', join(target, 'test', 'src', 'unittest'),
                                 join(target, 'test', 'unittest')])
    ret += subprocess.call(['rm', '-rf', join(target, 'test', 'src')])
    ret += subprocess.call(['rm', '-rf', join(target, 'test', 'SConscript')])

def chown(dir, owner, group, recursive=True):
    if recursive:
        return subprocess.call(['sudo', 'chown', '-R', owner + ':' + group, dir])
    return subprocess.call(['sudo', 'chown', owner + ':' + group, dir])

def removeDSSTORE(dir):
    # remove all .DS_Store files in directory
    ds_file = '.DS_Store'
    files = os.listdir(dir)
    for file in files:
        if os.path.isdir(file): removeDSSTORE(file)
        else:
            if file == ds_file: print "removing", file
    ret = 0
    #print 'files:', files
    #ret = 0
    #for file in files:
    #    print 'removing', file
    #    #ret += os.remove(file)
    return ret

def timeToString():
    t = time.localtime()
    month = '0'
    day = '0'
    year = str(t[0])
    if t[1] < 10: month += str(t[1])
    else: month = str(t[1])
    if t[2] < 10: day += str(t[2])
    else: day = str(t[2])
    return day+month+year


def buildPackage(pkg_dir, essentia_root):
    packageMaker = '/Developer/Applications/Utilities/PackageMaker.app/Contents/MacOS/PackageMaker'
    if not os.path.exists(packageMaker):
        print "Error building package. Could not find PackageMaker application"
        return 1

    root_dir = pkg_dir # path to where the distribution has been created
    python_version=sys.version[:3]
    if 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
        import platform
        os.environ['MACOSX_DEPLOYMENT_TARGET'] = '.'.join(platform.mac_ver()[0].split('.')[:2])
    target = os.environ['MACOSX_DEPLOYMENT_TARGET']

    # package name:
    vfh = open(join(essentia_root_dir,'VERSION'), mode='r')
    essentia_version = vfh.read().strip()
    vfh.close()
    title = 'essentia_v' + essentia_version + '_osx' + target + '_i386_python' + python_version
    domain = 'system' # where is going to be installed
    destination = '/' + title + '.pkg' # name to save the package
    id = title + '-' + timeToString() # unique id for package

    # path to scripts that will be run  either in pre/post install:
    scripts_path = join(essentia_root_dir, 'packaging', 'darwin', 'scripts')
    cmd = [ packageMaker,
           '--root', root_dir,
           '--title', title +'.pkg',
           '--version', essentia_version,
           '--target', target,
           '--domain', domain,
           '--out', destination,
           '--id', id,
           '--scripts', scripts_path]
    return subprocess.call(cmd)



if __name__ == '__main__':
    from optparse import OptionParser
    opt,args = OptionParser().parse_args()
    if (len(args) < 2):
        print 'error while finalising osx package: wrong number of arguments'
        print '\tusage: ./osx_finalise.py essentia_source_directory essentia_target_directory'
        sys.exit(1)

    build_dir = args[0]
    essentia_root_dir = "/".join(build_dir.split('/')[:-1])
    target_dir = args[1]
    print '*'*70
    print 'build_dir:', build_dir
    print 'essentia_dir:', essentia_root_dir
    print 'target_dir:', target_dir
    print '*'*70

    print '\n'
    print '*'*70
    print '* copying examples to', target_dir
    if copy_examples(essentia_root_dir,             # where essentia src lives
                     join(build_dir,'examples'),    # where examples are built
                     join(target_dir, 'examples')): # where we want to copy examples
        print "An error occurred while finalising osx package: error while\
        copying examples"
        sys.exit(1)

    print '* copying documentation to', target_dir
    if copy_documentation(essentia_root_dir, target_dir):
        print "An error occurred while finalising osx package: error while\
        copying documentation"
        sys.exit(2)

    print '* removing \".DS_Store\" files from', target_dir
    if removeDSSTORE(target_dir):
        print "An error occurred while finalising osx package: "\
              " could not delete all \".DS_Store\" files."
        sys.exit(3)

    print '* copying tests from', essentia_root_dir, 'to', target_dir
    if copy_python_tests(essentia_root_dir, target_dir):
        print "An error occurred while finalising osx package: "\
              " could not copy unittest files."
        sys.exit(3)

    pkg_dir = PKG_DIR
    print '* creating distribution pkg in', pkg_dir
    if subprocess.call(['mkdir', pkg_dir]):
        print "An error occurred while finalising osx package: could not "\
              "create", pkg_dir
        sys.exit(4)

    # store where usr and Library are before moving them:
    usr_dir = join(build_dir, 'packaging')
    library_dir = join(build_dir, 'Library')

    if subprocess.call(['mkdir', join(pkg_dir,'usr')]):
        print "An error occurred while finalising osx package: could not "\
              "create directory:", join(pkg_dir,'usr')

    if subprocess.call(['cp', '-r', usr_dir, join(pkg_dir,'usr', 'local')]):
        print "An error occurred while finalising osx package: could not "\
              "move", usr_dir, "to", pkg_dir
        sys.exit(5)

    if subprocess.call(['mv', library_dir, pkg_dir]):
        print "An error occurred while finalising osx package: could not "\
              "move", library_dir, "to", pkg_dir
        sys.exit(6)

    if subprocess.call(['mv', target_dir, pkg_dir]):
        print "An error occurred while finalising osx package: could not "\
              "copy ", target_dir, "to", pkg_dir
        sys.exit(7)


    # change ownership of usr folder to root:wheel:
    usr_dir = join(pkg_dir, 'usr')
    print '* changing ownership of ', usr_dir, 'to root:wheel'
    if chown(usr_dir, 'root', 'wheel', True):
        print "An error occurred while finalising osx package: "\
              "could not change ownership of \"usr\" folder"
        sys.exit(8)

    # change ownership of Library folder to root:admin:
    library_dir = join(pkg_dir, 'Library')
    print '* changing ownership of ', library_dir, 'to root:admin'
    if chown(library_dir, 'root', 'admin', True):
        print "An error occurred while finalising osx package: error while "\
              "changing ownership of \"Library\" folder"
        sys.exit(9)

    print "* Building Package..."
    if buildPackage(PKG_DIR, essentia_root_dir):
        print "An error occurred while finalising osx package: error while "\
              "creating the package"
        # osx packageMaker failed, but distribuiton package was done succesfully:
        print '\n'
        print "*"*70
        print "* osx distribution build successful. You may find it in", pkg_dir
        print "* in order to build the package you must use PackageMaker found in "\
              "* /Developer"
        print "*"*70
        print '\n'
        sys.exit(0)
    else:
        print '\n'
        print "*"*70
        print "* osx package build successful. You may find it in your root "\
              "directory"
        print "*"*70
        print '\n'
        sys.exit(0)

