import shutil, sys, importlib, os
patched_script = 'safety_checker_patched.py'

def get_python_modules_dir():
    # get where python modules are installed for every operating system
    if sys.platform == 'win32':
        python_version = "Python%d%d" % (sys.version_info.major, sys.version_info.minor)
        dir = os.path.expandvars(f"%LOCALAPPDATA%\\Packages\\")
        # search folder than starts with "PythonSoftwareFoundation.Python"
        for folder in os.listdir(dir):
            if folder.startswith("PythonSoftwareFoundation.Python"):
                return os.path.join(dir, folder, "LocalCache", "local-packages", python_version, "site-packages")
        return os.path.join(sys.prefix, 'Lib', 'site-packages')
    elif 'google.colab' in sys.modules:
        python_version = "python%d.%d" % (sys.version_info.major, sys.version_info.minor)
        return '/usr/local/lib/%s/dist-packages' % python_version
    else:
        python_version = "python%d.%d" % (sys.version_info.major, sys.version_info.minor)
        return os.path.join(sys.prefix, 'lib', python_version, 'site-packages')
def empty_safety_checker(images, **kwargs):
    return images, False
def try_patch(pipe):
    try:
        target_script = os.path.join(get_python_modules_dir(), 'diffusers', 'pipelines', 'stable_diffusion', 'safety_checker.py')
        if(os.path.exists(patched_script)):
            shutil.copyfile(patched_script, target_script)
            importlib.invalidate_caches()
        pipe.safety_checker = empty_safety_checker
        return True
    except Exception as e:
        print("Couldn't patch NSFW: ", e)
        return False