# Breast_Cancer_Diagnosis_V2
An enhanced model for breast cancer diagnosis (using transfer learning: Feature extractinn) using [TensroFlow](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetV2B0)'s `EfficientNetV2B0` model.

Where in the first model ( [Breast_Cancer_Diagnosis](https://github.com/waliddj/Breast_Cancer_Diagnosis) ) a convolutional neuron network was built from scratch. Breast_Cancer_Diagnosis_V2 model is based on
[TensroFlow](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetV2B0)'s `EfficientNetV2B0` model.

# Dataset
---

The dataset used for this model is the [Breast cancer dataset](https://www.kaggle.com/datasets/djaidwalid/)

*Citation:*: https://www.kaggle.com/datasets/djaidwalid/

---
## Dataset structure:
This dataset is divided into two main directories `train` and `test` directory each divided into two other directories `breast_malignant` and `breast_benign` following this structure:
```
\train
  |
  |__\breast_malignant
      |
      |__4000 images
  |
  |__\breast_benign
      |
      |__4000 images

\test
  |
  |__\breast_malignant
      |
      |__1000 images
  |
  |__\breast_benign
      |  
      |__1000 images

```
## Dataset details:

|Path|	Subclass|Description|
|-----|-----------|--------------|
|breast_benign|	Benign|	Non-cancerous (healthy) breast tissues|
|breast_malignant|	Malignant|	TCancerous breast tissues|

*Source: Collected from the Breast Cancer dataset by Anas Elmasry on Kaggle.*

## Data augmentation:
the data was augmentend by the original author of the dataset using Kera's `ImageDataGeberator` *[1]* and The augmentations include:
-  Rotation: Up to 10 degrees.
-   Width & Height Shift: Up to 10% of the total image size.
-   Shearing & Zooming: 10% variation.
-   Horizontal Flip: Randomly flips images for additional diversity.
-   Brightness Adjustment: Ranges from 0.2 to 1.2 for varying light conditions.

The parameters used for augmentation:
```python
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.2, 1.2]
)

```
*For more details visit the [Breast cancer dataset]() kaggle page* 

# Code architecture:
## Dataset
### Load the dataset from directory
```python
data_dir = 'C:/Users/walid/.cache/kagglehub/datasets/obulisainaren/multi-cancer/versions/3/Multi Cancer/Multi Cancer/Breast Cancer/'
```
### Split the training and test data directories
```python
train_dir = 'C:/Users/walid/.cache/kagglehub/datasets/obulisainaren/multi-cancer/versions/3/Multi Cancer/Multi Cancer/Breast Cancer/train'
test_dir = 'C:/Users/walid/.cache/kagglehub/datasets/obulisainaren/multi-cancer/versions/3/Multi Cancer/Multi Cancer/Breast Cancer/test'
```
### Split the training and test data using `tf.keras.preprocessing.image_dataset_from_directory`
```python
IMAGE_SIZE = (224,224) # Fix the image sizes of the train and test data
BATCH_SIZE= 32 # fix the data batch size

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size= IMAGE_SIZE,
    label_mode='binary',
    batch_size=BATCH_SIZE,
)
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size= IMAGE_SIZE,
    label_mode='binary'
)
```
### Get the class names (labels) from the `train_data`
```python
class_names = train_data.class_names
```
## Build the model *(Transfer learning : feater extraction)*:
### Create checkpoint callbacks:
```python
checkpoint_path = 'C:/Users/walid/.cache/kagglehub/datasets/obulisainaren/multi-cancer/versions/3/Multi Cancer/Multi Cancer/Breast Cancer/checkpoint.weights.h5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                save_weights_only=True,
                                                save_best_only= True,
                                                save_freq='epoch',
                                                verbose=1)
```

### Create a base line model
**Steps:**
1. Create a base_line model using `tf.keras.applications.efficientnet_v2.EfficientNetV2B0`.
2.  Freeze the base model (so the pre-learned patterns remain).
```python
base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top = False)
base_model.trainable = False
```
### Create inputs into the base model
```python
inputs = tf.keras.layers.Input(shape=(224,224,3), name='input_layer')
### Pass the inputs to the base_model (note: using tf.keras.applications, EfficientNetV2 inputs don't have to be normalized)
x = base_model(inputs)
```

###  Average pool the outputs of the base model (aggregate all the most important information, reduce number of computations)
```python
x = tf.keras.layers.GlobalAveragePooling2D(name='pool_layer')(x)
```
### Create the output activation layer
```python
outputs= tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')(x)
```

### Combine the inputs with the outputs into a model
```python
model = tf.keras.Model(inputs,outputs)
```
### Compile the model
```python
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = 'adam',
    metrics=['accuracy']
)
```
### Fit the model
```python
initial_epoch = 4
history_1 = model.fit(train_data,
                      epochs = initial_epoch,
                      steps_per_epoch = len(train_data),
                      validation_data=test_data,
                      validation_steps = (test_data),
                      callbacks=[checkpoint])
```
# Evaluate the model
## Loss&Accuracy
|Accuracy|Loss|
|--------|----|
|99.31%|0.0272|

## Confusion matrix
<img width="708" height="480" alt="cm_1" src="https://github.com/user-attachments/assets/f29f1c09-05e9-4cb6-916d-3710fbaa586f" />

## Predictions
<img width="420" height="420" alt="pred_4" src="https://github.com/user-attachments/assets/eca9939b-e70e-4230-9801-169033e94bc5" />
<img width="420" height="420" alt="pred_3" src="https://github.com/user-attachments/assets/7fbbff15-d3f8-4ad0-92e9-cccb30983765" />
<img width="420" height="420" alt="pred_2" src="https://github.com/user-attachments/assets/47c957cf-fd22-4dc2-a655-3bc08ca61907" />
<img width="420" height="420" alt="pred_1" src="https://github.com/user-attachments/assets/9ca49151-47c7-468a-a703-eb867d07cb86" />
<img width="420" height="420" alt="pred_0" src="https://github.com/user-attachments/assets/65a715e8-46e4-406c-9a69-7baf427b9915" />


# Save the model
```python
model.save("C:/Users/walid/Desktop/Breast_Cancer_2.keras")
```
# [`Breast_Cancer_Diagnosis`](https://github.com/waliddj/Breast_Cancer_Diagnosis) vs `Breast_Cancer_Diagnosis_V2`

## Loss&Accuracy

|Model|Accuracy|Loss|
|--|--|--|
|`Breast_Cancer_Diagnosis`|94.20% |0.161 |
|`Breast_Cancer_Diagnosis_V2`|**99.31%** |**0.0272**|

> the `Breast_Cancer_Diagnosis_V2` 5% more accurate. Moreover, its loss function is 4 times lower.

## Confusion matrix
<img width="1349" height="549" alt="cm1_cm2" src="https://github.com/user-attachments/assets/149618bc-2527-4a5c-a1b8-8308b9d8e1b5" />

> The confusion matrix proves that again `Breast_Cancer_Diagnosis_V2` is more accurate than `Breast_Cancer_Diagnosis`.

