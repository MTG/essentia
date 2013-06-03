#!/usr/bin/python

import glob
import subprocess

def add_to(var, options):
    for platform in options:
        # do we have a default value that we need to put?
        if platform == 'default' and sys.platform not in options:
            if type(options[platform]) is not list:
                var += [ options[platform] ]
            else:
                var += options[platform]
            return

        # do we have a specific value for this platform?
        if platform == sys.platform:
            if type(options[platform]) is not list:
                var += [ options[platform] ]
            else:
                var += options[platform]
            return


def gcc_version():
    full_version = os.popen('g++ --version | head -1').read()
    return full_version.split(' ')[2]


def change_root(f, new_root):
    try:
        if f.startswith('/'):
            return f
        return new_root + '/' + f
    except:
        return f

def set_root(env, new_root):
    env['CPPPATH'] = map(lambda x: change_root(x, new_root), env['CPPPATH'])
    env['LIBPATH'] = map(lambda x: change_root(x, new_root), env['LIBPATH'])


def get_flag(name, default=True):
    return ARGUMENTS.get(name, str(default)).lower() == 'true'

def replace_last(str, substr1, substr2):
    return (str[::-1].replace(substr1[::-1], substr2[::-1], 1))[::-1]

def any_of(namelist, names):
    for name in names:
        if name in namelist:
            return True
    return False


def files_in_dir(*args):
    return glob.glob(apply(os.path.join, args))

def fetch_env_variable(name, required = True):
    if name not in os.environ:
        if required:
            raise EnvironmentError('Variable \'%s\' is not defined in your environment' % name)
        else:
            return '%s = \'\'' % name

    return '%s = os.environ[\'%s\']' % (name, name)


def install_files(source, ext, dest):
    for root, dirs, files in os.walk(source):
        if '.svn' not in root:
            for f in files:
                if f.endswith(ext):
                    env.Install(dest, os.path.join(root, f))


Export('files_in_dir', 'install_files')

class BuildConfig:
    pass

def build_sh_cmd(env, id, executable, args=[], exec_dir=None, local=True):
    if not isinstance(args,list): raise TypeError('\'args\' argument must be a list')

    def ShellCmd(target, source, env):
        if local: exe = join(os.getcwd(),executable)
        else: exe = executable
        msg = 'Running command: \''+exe+' '+' '.join(args)+'\''
        if exec_dir != None: msg += ' from \''+exec_dir+'\''
        print msg
        proc = subprocess.Popen([exe]+args, stdout=subprocess.PIPE, cwd=exec_dir)
        print proc.communicate()[0]
        return proc.returncode

    return env.Command(id, [], ShellCmd)

Export('build_sh_cmd')

def removeMacPortsFromPath(paths):
    print 'Removing Macports includes from path...', paths
    # get rid of any /opt/ (macports) in env['ENV']['PATH']
    #env['ENV']['PATH'] = [ path for path in env['ENV']['PATH'] i '/opt' not in path]
    if type(paths) == str:
      paths = paths.split(':')
      lpath = ''
      paths = [ path for path in paths if '/opt' not in path]
      for path in paths[:-1]:
          lpath += path + ':'
      lpath += paths[-1]
    else:
        lpath = [ path for path in paths if '/opt' not in path]
    return lpath


Export('removeMacPortsFromPath')
