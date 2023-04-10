import os, sys, IPython
def install_vendors():
    
    print("Installing vendors -> ", end="")
    import os, IPython
    if os.path.exists("../vendor/.installed"):
        print("Done.")
        return
    try:
        f = open(os.devnull, 'w')
        original_stdout = sys.stdout
        sys.stdout = f
        os.mkdir("../vendor")
        # GFPGAN
        os.remove("../vendor/GFPGAN") if os.path.exists("../vendor/GFPGAN") else None
        # git clone using IPython magic
        IPython.get_ipython().system("git clone https://github.com/TencentARC/GFPGAN.git vendor/GFPGAN > /dev/null")
        os.chdir("../vendor/GFPGAN")
        IPython.get_ipython().system("pip install basicsr > /dev/null")
        IPython.get_ipython().system("pip install facexlib > /dev/null")
        IPython.get_ipython().system("pip install -q -r requirements.txt > /dev/null")
        IPython.get_ipython().system("python setup.py develop > /dev/null")
        # used for enhancing the background (non-face) regions
        IPython.get_ipython().system("pip install realesrgan > /dev/null")
        # used for enhancing the background (non-face) regions
        IPython.get_ipython().system("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.8/GFPGANv1.3.pth -P experiments/pretrained_models > /dev/null")
        os.chdir("../../..")
        # Use vendor/GFPGAN/inference_gfpgan.py to download & cache the model
        if not os.path.exists("../vendor/GFPGAN/results/whole_imgs"):
            IPython.get_ipython().system("python vendor/GFPGAN/inference_gfpgan.py -i vendor/GFPGAN/inputs/whole_imgs -o vendor/GFPGAN/results/whole_imgs -v 1.3 -s 2 --bg_upsampler realesrgan > /dev/null 2>&1")
        # Real-ESRGAN
        os.remove("../vendor/Real-ESRGAN") if os.path.exists("../vendor/Real-ESRGAN") else None
        IPython.get_ipython().system("git clone https://github.com/xinntao/Real-ESRGAN.git vendor/Real-ESRGAN > /dev/null")
        os.chdir("../vendor/Real-ESRGAN")
        IPython.get_ipython().system("wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P weights > /dev/null")
        IPython.get_ipython().system("wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -p weights > /dev/null")
        IPython.get_ipython().system("pip install basicsr facexlib gfpgan ffmpeg-python > /dev/null")
        IPython.get_ipython().system("pip install -q -r requirements.txt > /dev/null")
        IPython.get_ipython().system("python setup.py develop > /dev/null")
        os.chdir("../../..")
        # generate a random file to mark that the vendors are installed
        open("../vendor/.installed", "w").close()
        # StyleGAN2
        sys.stdout = original_stdout
        print("Done.")
    except Exception as e:
        print("Error: %s" % e)
