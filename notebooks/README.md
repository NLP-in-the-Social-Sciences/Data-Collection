# Installation Instructions
***
To start, I recommend using Python 3.9 as it has causes us the least amount of dependency issues.

For venv use the following command to create your environment using venv: 
```bash
python -m venv .venv
```
or, for conda use the following: 
```bash 
conda create -n environment_name python=3.9
```

## General (for all of this directory) 
Make sure to install Microsoft Visual C++ 14.0 or greater, Windows SDK for your specific OS and .NET 7.0 SDK (latest). Get it here with [Microsoft C++ Build Tools][Build tools] . Check the boxes for each one and install them. 


Additionally, make sure to install the CUDA dev kit (<u>**only version 11.7 or 11.8**</u>) from [here][CUDA] and install the latest kit. And then install cuda-python using the following command: 
```bash
pip install cuda-python
``` 
Then install torch using the command below for windows which will install the cuda 11.8 compatible version: 
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
``` 
Otherwise, goto [pytorch's installation page][pytorch installation] to specify your download parameters. 

## Filtering Search Engine
There are two versions of this search engine: one based on keywords extracted from the handpicked stories that and the other based on embeddings of the narratives themselves. To be continued...

[Build tools]: https://visualstudio.microsoft.com/visual-cpp-build-tools/
[CUDA]: https://developer.nvidia.com/cuda-11-8-0-download-archive
[pytorch installation]: https://pytorch.org/get-started/locally/