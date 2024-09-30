Use dynamic python 3.11 for tracy:

```bash
PYTHON_CONFIGURE_OPTS='--enable-shared' LDFLAGS="-Wl,-rpath=$HOME/.pyenv/versions/3.11.10/lib" pyenv install 3.11.10
```

Use nightly torch

```
# requirements.txt
--pre --index-url https://download.pytorch.org/whl/nightly/cu124
torch
torchvision
torchaudio
```

Build patched tracy with pyenv

```bash
# Build c++ library
mkdir build && cd build
cmake -DTRACY_STATIC=OFF -DTRACY_CLIENT_PYTHON=ON -DPython_FIND_VIRTUALENV=ONLY ..
make -j8

# Install python binding
cd ../python
python3 setup.py install

# build server library
cmake -B profiler/build -S profiler -DCMAKE_BUILD_TYPE=Release -DLEGACY=ON -DGTK_FILESELECTOR=ON
cmake --build profiler/build --config Release --parallel
```
