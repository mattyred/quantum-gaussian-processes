import os
import shutil

def setup_latex_environment():
    if shutil.which('latex'):
        os.environ['PATH'] = "/sw/apps/texlive/2022/bin/x86_64-linux:" + os.environ['PATH']
        # Tell TeX to use a local cache for font generation to avoid permission issues
        os.environ['TEXMFVAR'] = os.path.expanduser('~/.cache/texmf')