## What is ID-System?

ID-System is a simple tk GUI with 2 functionalities : 

 1. Checking similarity between ID photo and DB embeddings
    
 3. Checking Real Time captured picture similarity with ID photo (if ID photo embeddings 'True' in the DB) 

## Packages :

 **`CUDA 11.3`** **`Python 3.9.7`** **`scipy 1.11.1`** **`tqdm 4.65.0`** **`conda 23.5.0`**   **`docker 6.1.3`**

*There are much more packages, please check* **`requirements.txt`**.



## Used Models and Tools :

 1. `Python tk` is used to make user GUI
    
 3. `RetinaFace` - model to Face Detection task
    
 5. `ArcFace` - model to Face Recognition
    
 7. `Google Real Time DB`  - database used to save photo embeddings.
    
 9. `Single-GPU RTX 3060` - used to train models 

*Note : All models and Tools are used with PyTorch (Python) implementation.*



## How `.py` files work ?

 - There are 2 main user GUI python files: **`real_time_checker_gui.py`** and **`id_checker_gui.py`**
 
 - **`Core_detector.py`** is the main python file with Retinaface implementation to face detection
 
 - **`Core_recognizer.py`** is the another python file with Arcface implementation to face recognition task of Computer Vision.
 
 - **`subprocess_files.py`** and **`subprocess_real_time_files.py`** files are responsible to connect multiple python files.
 
 - **`server_connect.py`** file helps to connect to DB (Google Real  time).

- **`ID_card_delete_resources.py`** and **`delete_files_real_time.py`** files are responsible to delete all unnecessary img and npy files. (recommended to run every time)
   
 *Note: Some codes in python files may not work or (private information) may be missed (deleted on purpose bcz of privacy. )*

## Folders :

 - **`conv_npy`** , **`data, ID_picure,`** **`Real_time_image`** and **`GUI_src`** are folders with bunch of images (*id, user real time captured image, etc.*) and .npy (*db saved numpy embeddings or some ArcFace calculation .npy files*)
 
 - **`Detector`** folder consists of resources of *RetinaFace FD* codes.
 - **`Recognizer`** is another folder with *Arcface FR* task modules and codes.

 *Note: Some folders  may not work or (private information) may be missed (deleted on purpose bcz of privacy. )*

## Workflow map :





## Resources :

 - [RetinaFace Pytorch Implementation](https://github.com/serengil/retinaface)

 - [ArcFace Pytorch Implementation](https://github.com/deepinsight/insightface)
   
 - [ChatGPT to search and fix bugs](https://chat.openai.com/)
   
 - [Face Anti Spoofing Silent Face](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
   
 - [Google Real Time Database](https://firebase.google.com/)

## Caution :

    
Note : Most model implementation codes are open-source but have copyrights to certain authors, so I decided not to delete authors'

names in codes. In the future projects of Face Recognition, you can use that codes with modifications.  I trained RetinaFace and 

ArcFace with my `custom dataset + WiderFace(RetinaFace ) and CelebA(ArcFace)`.  So, I can't provide models and weights directly 

(privacy issues). If you want to use them, please contact with me first **`uacoding01@gmail.com`**.



## Thank you...


  
 



