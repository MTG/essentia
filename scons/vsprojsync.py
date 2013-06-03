# update a visual studio project with a list of files
# (c) 2006 Ross Bencina <rossb@audiomulch.com>

import xml.dom.minidom
import os.path
import sys

def getAllFilesRelativePaths( domDocument ):
    result = []
    fileElements = domDocument.getElementsByTagName( "File" )
    for e in fileElements:
        result.append( e.getAttribute("RelativePath") )
        
    return result

def fileListsAreEqual( a, b ):
    for s in a:
        if len(s) == 0:
            continue
        if not s in b:
            return 0
    for s in b:
        if len(s) == 0:
            continue
        if not s in a:
            return 0
    return 1

def removeDirectoryElements( domDocument ):
    fileElements = domDocument.getElementsByTagName( "Filter" )
    for e in fileElements:
        if e.hasAttribute("Name"):
            e.parentNode.removeChild( e )

def removeFileElements( domDocument ):
    fileElements = domDocument.getElementsByTagName( "File" )
    for e in fileElements:
        e.parentNode.removeChild( e )

def createFileNode( domDocument, relativePath ):
    result = domDocument.createElement( "File" )
    result.setAttribute( "RelativePath", relativePath )
    result.appendChild( domDocument.createTextNode("") )
    return result

def createDirectoryNode( domDocument, name ):
    result = domDocument.createElement( "Filter" )
    result.setAttribute( "Name", name )
    result.appendChild( domDocument.createTextNode("") )
    return result

def insertFilesIntoProject( domDocument, filesElement, filesToInsert ):
    for f in filesToInsert:
        (directory, filename) = os.path.split(os.path.normpath(f))
        dom_element = filesElement
        
        for d in directory.replace("\\", "/").split("/"):
            chosen_child = None
            # see if we have a child node with the correct name/attributes
            for child in dom_element.childNodes:
                if child.localName == "Filter" and child.hasAttribute("Name") and child.getAttribute("Name") == d:
                    chosen_child = child
            # if this child node doesn't exist, create it
            if not chosen_child:
                chosen_child = createDirectoryNode(domDocument, d)
                dom_element.appendChild( chosen_child )
            # recurse
            dom_element = chosen_child
                
        dom_element.appendChild( createFileNode( domDocument, f ) )


def replaceProjectFiles( domDocument, newFiles ):
    removeFileElements( domDocument )
    removeDirectoryElements( domDocument )
    filesElement = domDocument.getElementsByTagName( "Files" )[0] # assume there's only one <Files> tag
    insertFilesIntoProject( domDocument, filesElement, newFiles )

def includePathsAreEqual( domDocument, newIncludePaths ):
    toolElements = domDocument.getElementsByTagName( "Tool" )
    for e in toolElements:
        includeSearchPath = e.getAttribute("IncludeSearchPath").split(";")
        if not fileListsAreEqual( includeSearchPath, newIncludePaths ):
            #print( includeSearchPath )
            #print( newIncludePaths );
            return 0
    return 1

def replaceIncludePaths( domDocument, newIncludePaths ):
    includePathString = ""
    for p in newIncludePaths:
        includePathString += ";" + p
    toolElements = domDocument.getElementsByTagName( "Tool" )
    for e in toolElements:
        e.setAttribute("IncludeSearchPath", includePathString )

def updateVCProjectFileFileList( projectFileName, newFilesList, newIncludePaths, force=0 ):
    updated = 0
    vcprojDom = xml.dom.minidom.parse(projectFileName)
    currentProjectFiles = getAllFilesRelativePaths(vcprojDom)
    if force or not fileListsAreEqual( newFilesList, currentProjectFiles ):
        # print( "updating project files" )
        replaceProjectFiles( vcprojDom, newFilesList )
        updated = 1

    if force or not includePathsAreEqual( vcprojDom, newIncludePaths ):
        # print( "updating include paths" )
        replaceIncludePaths( vcprojDom, newIncludePaths )
        updated = 1

    if updated:
        vcprojDom.writexml( file(projectFileName,"w") )
        
    #vcprojDom.writexml( sys.stdout )
    return updated

##projectFileName = "C:\\amtrunk_wd\\Win32\\MSVCPortingStubs\\MSVCMulch\\MSVCMulch\\MSVCMulch.vcproj"
##
##testFiles = [ "x.txt", "y.cpp", "d.h", "ggg.res" ]
##
##updateVCProjectFileFileList( projectFileName, testFiles )
##
