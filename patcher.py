import shutil, sys
patch_path = '/content/void-diffusion/safety_checker_patched.py'
def patch():
    python_version = "python%d.%d" % (sys.version_info.major, sys.version_info.minor)
    target_path = '/usr/local/lib/%s/dist-packages/diffusers/pipelines/stable_diffusion/safety_checker.py' % python_version
    shutil.copyfile(patch_path, target_path)