Please use Python 3.8 for this project to avoid getting package conflict

If you are using Anaconda, please install the following dependencies for running the code without errors and warnings:
```
conda install m2w64-toolchain
```

Please add a file called ".theanorc" in your home directory with the content shown as below:
```
[blas]
ldflags = -lmkl_rt
```

optional dependencies for windows and conda, which is related in intel cpu processor
```
conda install mkl
conda install mkl-service
conda install blas
```



Run this command to reinstall the packages for refreshing your environment
```bash
pip install --force-reinstall -r requirements.txt
```

If you encounter compiling errors while using the model NetMF with the conda environments:

Please run the following command for conda installing a mingw-w64 import library for python

```bash
conda install -c anaconda libpython
```
