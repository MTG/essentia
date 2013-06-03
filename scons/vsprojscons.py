"""SCons.Tool.vsprojscons

    update a visual studio project with a list of files
    (c) 2006 Ross Bencina <rossb@audiomulch.com>

"""

import vsprojsync
import os.path

import SCons.Action
import SCons.Builder
import SCons.Defaults
import SCons.Scanner
import SCons.Tool
import SCons.Util


def vcprojsync_function(target, source, env):
    # at the moment we use a hack here so that scons doesn't delete the target
    vcProjectFileName = os.path.normpath(str(target[0]))

    #print(vcProjectFileName)
    
    # our path code can't handle placing the visual studio project
    # above the SConstruct file in the file tree
    if os.path.isabs( vcProjectFileName ):
        print "ERROR: visual studio project must be inside project tree"
        
    # scons file names are relative to the project root directory
    # but the vs project needs the paths relative to its directory
    # so we calculate the relative prefix to use here:
    relativePrefix = ""
    for i in range(len(vcProjectFileName.split("\\")) - 1):
        relativePrefix += "..\\"

    #print relativePrefix

    # collect all source file names and also all implicit
    # dependencies (header files)
    files = []
    for o in source:
        for s in o.sources:
            ss = relativePrefix + str(s)
            if ss not in files:
                files.append(ss)
        if o.implicit:
            for d in o.implicit:
                sd = relativePrefix + str(d)
                if sd not in files:
                    files.append(sd)

    #TODO: update include paths in vcproj from dependent headers
    force = 0 # set this to 1 if the file list has to be updated
    if vsprojsync.updateVCProjectFileFileList( vcProjectFileName, files, env['CPPPATH'], force ):
        print "Visual Studio project file list updated."

    return None


def generate(env):
    # install the visual studio synchronisation builder
    env['BUILDERS']['SyncVCProj'] = SCons.Builder.Builder(action = SCons.Action.Action(vcprojsync_function, strfunction=None))

def exists(env):
    return True
