## What is ID-System?

ID-System is a simple tk GUI with 2 functionalities : 

 1. Checking similarity between ID photo and DB embeddings
 2. Checking Real Time captured picture similarity with ID photo (if ID photo embeddings 'True' in the DB) 

## Packages :

 **`CUDA 11.3`** **`Python 3.9.7`** **`scipy 1.11.1`** **`tqdm 4.65.0`** **`conda 23.5.0`**   **`docker 6.1.3`**

*There are much more packages, please check* **`requirements.txt`**.



## Used Models and Tools :

 1. `Python tk` is used to make user GUI
 2. `RetinaFace` - model to Face Detection task
 3. `ArcFace` - model to Face Recognition
 4. `Google Real Time DB`  - database used to save photo embeddings. 
 5. `Single-GPU RTX 3060` - used to train models 

*Note : All models and Tools are used with PyTorch (Python) implementation.*



## How it works ?

 - There are 2 main user GUI python files: **`real_time_checker_gui.py`** and **`id_checker_gui.py`**
