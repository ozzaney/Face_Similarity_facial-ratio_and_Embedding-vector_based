# Face_Similarity_facial-ratio_and_Embedding-vector_based

# Categorical-Time-Series-Prediction-with-Embeddings
![image]([https://user-images.githubusercontent.com/87710236/163808952-e3c93f9e-2f47-4373-b624-bde02608babf.png](https://github.com/jayl-ee/Face_Similarity_facial-ratio_and_Embedding-vector_based/blob/main/Data/스크린샷%202022-05-28%20오후%2012.01.56.png))

플랫폼 내에서 고객 행동 데이터를 기반으로 다음 행동을 예측하여 Personalized한 서비스 제공할 수 있는 기회 제공을 목적으로 하며, L.POINT 데이터를 활용하였다.
### DSL 기업연계 프로젝트 (2022. 05)
___

### Collaborators
이승재, 황다연, 손예진, 전재현, 이승연, 이승주
___
### Requirements
numpy>=1.14.0
pandas>=0.23.4
gdown>=3.10.1
tqdm>=4.30.0
Pillow>=5.2.0
opencv-python>=4.5.5.64
opencv-contrib-python>=4.3.0.36
tensorflow>=1.9.0
keras>=2.2.0
Flask>=1.1.2
mtcnn>=0.1.0
lightgbm>=2.3.1
dlib>=19.20.0
retina-face>=0.0.1
mediapipe>=0.8.7.3
fire>=0.4.0

## File Definition
### 📁code
| Data
| Model -- deepface
             -- * deepface -- basemodels
                           -- commons
                           -- weight
                           -- preprocessing.ipynb
                           -- DeepFace.py
                           -- get_ratio.py
                           -- get_ratio_side.py

        -- penultimate_layer
            -- Attention
            -- Tree-based
            -- Vers


___
### General Process of the Model
* - Crop image to concentrate on the face
* - Get embedding layer for the image
* - Get facial ratio of the image
* - input two variables to the "penultimate layer"
* - Classification task for predicting similarity class

```python
    fron deepface.deepface import DeepFace as DF
    from deepface.deepface import get_ratio, get_ratio_side

    #get embedding vector
    DF.represent( IMG_PATH )

    #get facial ratio
    get_ratio.get_ratio( IMG_PATH )

    

___

# Reference
- https://github.com/serengil/deepface.git
