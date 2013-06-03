# name of program
Name 'Essentia'

!define ESSENTIA_ROOT '..\..'
!ifndef THIRDPARTY
#    !define THIRDPARTY 'C:\essentia-thirdparty-msvc2005sp1-32bit'
    !define THIRDPARTY 'C:\Documents and Settings\buildbot\build-space\new-essentia-thirdparty'
!endif
!ifndef FFMPEG_DIR
    !define FFMPEG_DIR 'C:\Documents and Settings\buildbot\build-space\ffmpeg0.5_install'
!endif

# installer filename
OutFile '${ESSENTIA_ROOT}\build\packaging\install_essentia.exe'

# default installation directory (user will be able to change this)
InstallDir '$PROGRAMFILES\Essentia\'

# enable xp styles
XPStyle on

Page components
Page directory
Page instfiles

UninstPage uninstConfirm
UninstPage instfiles

!include 'uninstall_ex_begin.nsh'
Section 'Prerequisites'
  SetOutPath $INSTDIR\Prerequisites
  MessageBox MB_YESNO "Install Microsoft Visual Studio 2005 SP 1 C-Runtime?" /SD IDYES IDNO endRedist
    File "C:\Program Files\Microsoft Visual Studio 8\SDK\v2.0\BootStrapper\Packages\vcredist_x86\vcredist_x86.exe"
    ExecWait "$INSTDIR\Prerequisites\vcredist_x86.exe"
    Goto endRedist
  endRedist:
SectionEnd

# everything that comes with the base
Section 'Essentia Library'
    ${SetOutPath} '$INSTDIR' # just need to set it once, is saved for the rest

    # create the uninstaller
    ${WriteUninstaller} 'uninstall.exe'

    # install the lib file
    ${SetOutPath} '$INSTDIR\lib'
    ${File} '${ESSENTIA_ROOT}\build\packaging\lib\' 'essentia.lib'

    # install the header files
    ${SetOutPath} '$INSTDIR\include'
    ${File} '${ESSENTIA_ROOT}\build\packaging\include\' '*.h'
    ${SetOutPath} '$INSTDIR\include\tnt'
    ${File} '${ESSENTIA_ROOT}\build\packaging\include\tnt\' '*.h'

    # install thirdparty libraries
    ${SetOutPath} '$INSTDIR\thirdparty'

    # install thirdparty binaries
    ${SetOutPath} '$INSTDIR\thirdparty\bin'
    ${File} '${THIRDPARTY}\bin\' '*.dll'

    # install thirdparty includes
    ${SetOutPath} '$INSTDIR\thirdparty\include'
    ${File} '${THIRDPARTY}\include\' '*.h'
    !macro include_sub_folder dirname
        ${SetOutPath} '$INSTDIR\thirdparty\include\${dirname}'
        ${File} '${THIRDPARTY}\include\${dirname}\' '*.h'
    !macroend
    #!insertmacro include_sub_folder 'libavcodec'
    #!insertmacro include_sub_folder 'libavformat'
    #!insertmacro include_sub_folder 'libavutil'
    !insertmacro include_sub_folder 'taglib'
    !insertmacro include_sub_folder 'tbb'

    # special case for ffmpeg (we don't need that, right?)
    #!macro include_sub_folder2 dirname
    #    ${SetOutPath} '$INSTDIR\thirdparty\include\${dirname}'
    #    ${File} '${FFMPEG_DIR}\include\${dirname}\' '*.h'
    #!macroend
    !insertmacro include_sub_folder 'libavcodec'
    !insertmacro include_sub_folder 'libavformat'
    !insertmacro include_sub_folder 'libavutil'

    # special case for tbb (because has a subfolder inside)
    ${SetOutPath} '$INSTDIR\thirdparty\include\tbb\machine'
    ${File} '${THIRDPARTY}\include\tbb\machine\' '*.h'

    # install thirdparty libs
    ${SetOutPath} '$INSTDIR\thirdparty\lib'
    ${File} '${THIRDPARTY}\lib\' '*.lib'

SectionEnd


Section 'Extractors'
    ${SetOutPath} '$INSTDIR\extractors'
    ${File} '${ESSENTIA_ROOT}\build\examples\' 'streaming_extractor.exe'
SectionEnd


Section 'Doxygen Documentation'
    ${SetOutPath} '$INSTDIR\doc'
    # we have to list each extension explicitly because of a limitation in the
    # auto-uninstall part of this script
    ${File} '${ESSENTIA_ROOT}\build\doc\doxygen\html\' '*.css'
    ${File} '${ESSENTIA_ROOT}\build\doc\doxygen\html\' '*.gif'
    ${File} '${ESSENTIA_ROOT}\build\doc\doxygen\html\' '*.png'
    ${File} '${ESSENTIA_ROOT}\build\doc\doxygen\html\' '*.html'
SectionEnd


!include 'envvarupdate.nsh'

Section 'Update PATH Environment Variable'
    # create new custom path key ESSENTIA_THIRDPARTY
    WriteRegStr HKLM 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment' \
                     'ESSENTIA_THIRDPARTY' '$INSTDIR\thirdparty\bin'

    #   append ESSENTIA_THIRDPARTY to path
    ${EnvVarUpdate} $0 'PATH' 'A' 'HKLM' '%ESSENTIA_THIRDPARTY%'
SectionEnd


# just files for the python bindings
Section 'Python Bindings'
    # Read latest version of python runtime
    # $0 - index
    # $1 - key-name
    StrCpy $0 0
    StrCpy $2 ''
    loop:
        EnumRegKey $1 HKLM 'Software\Python\PythonCore' $0
        StrCmp $1 '' done # if no more keys, goto done
        StrCpy $2 $1
        StrCmp $2 '2.7' valid_version # only looking for python 2.7
        IntOp $0 $0 + 1
        goto loop
    done:


    # $2 should contain latest version, or empty string (if python not installed)
    StrCmp $2 '2.7' valid_version

    # not valid version
    MessageBox MB_OK 'Valid Python installation not detected.$\n\
                      Please install Python version 2.7 and restart \
                      installation: http://python.org/download/'
    Abort 'Valid Python installation not detected. Please install Python \
           version 2.7: http://python.org/download/'

    valid_version:

    # read instalation directory into $0 (path will usually have trailing slash)
    ReadRegStr $0 HKLM 'Software\Python\PythonCore\$2\InstallPath' ''

    # gather python-binding related files and install
    ${SetOutPath} '$0Lib\site-packages\essentia'
    ${File} '${ESSENTIA_ROOT}\build\Python27\Lib\site-packages\essentia\' '_essentia.pyd'
    ${File} '${ESSENTIA_ROOT}\build\Python27\Lib\site-packages\essentia\' '*.py'
    ${File} '${ESSENTIA_ROOT}\build\Python27\Lib\site-packages\essentia\' '*.dll'

    # install the python extractor
    #${SetOutPath} '$INSTDIR'
    #${CreateDirectory} 'bin'
    #${SetOutPath} '$INSTDIR\bin'
    #${File} '${ESSENTIA_ROOT}\src\python\' 'essentia_extractor'

    # save essentia install location
    WriteRegStr HKLM 'Software\MTG\Essentia' 'PythonInstallDir' '$0Lib\site-packages\essentia'
SectionEnd


Section un.ExtraCleanup
    # delete any pyc files
    ReadRegStr $0 HKLM 'Software\MTG\Essentia' 'PythonInstallDir'
    Delete '$0\*.pyc'

    # clean up environment variables
    DeleteRegValue HKLM 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment' \
                        'ESSENTIA_THIRDPARTY'
    ${un.EnvVarUpdate} $0 'PATH' 'R' 'HKLM' '%ESSENTIA_THIRDPARTY%'

    # clean up other keys
    DeleteRegKey HKLM 'Software\MTG\Essentia'

    # if MTG now empty, clean up MTG key
    EnumRegKey $0 HKLM 'Software\MTG' 0
    StrCmp $0 "" +1 +2
    DeleteRegKey HKLM 'Software\MTG'

SectionEnd

!include 'uninstall_ex_end.nsh'
