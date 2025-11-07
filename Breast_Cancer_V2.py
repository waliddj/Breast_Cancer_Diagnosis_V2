"""
*****************************************************************************************
This model was built by: Djaid Walid

__________________________________________________________________________________________________
                                   Contacts                                                      |
__________________________________________________________________________________________________
Github     | https://github.com/waliddj                                                          |
Linkedin   | www.linkedin.com/in/walid-djaid-375777229                                           |
Instagram  | https://www.instagram.com/d.w.science?igsh=MWlnMmNpOTM2OW0xaA%3D%3D&utm_source=qr   |
__________________________________________________________________________________________________

*****************************************************************************************
"""
import tensorflow as tf
import matplotlib.pyplot as plt
data_dir = 'C:/Users/walid/.cache/kagglehub/datasets/obulisainaren/multi-cancer/versions/3/Multi Cancer/Multi Cancer/Breast Cancer/'
train_dir='C:/Users/walid/.cache/kagglehub/datasets/obulisainaren/multi-cancer/versions/3/Multi Cancer/Multi Cancer/Breast Cancer/train'
test_dir = 'C:/Users/walid/.cache/kagglehub/datasets/obulisainaren/multi-cancer/versions/3/Multi Cancer/Multi Cancer/Breast Cancer/test'

IMAGE_SIZE = (224,224)

train_data = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                  image_size = IMAGE_SIZE,
                                                                  batch_size=32,
                                                                  label_mode='binary')
test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                               image_size = IMAGE_SIZE,
                                                               label_mode='binary')
class_names = train_data.class_names

# Create checkpoint callbacks:
checkpoint_path = 'C:/Users/walid/.cache/kagglehub/datasets/obulisainaren/multi-cancer/versions/3/Multi Cancer/Multi Cancer/Breast Cancer/checkpoint.weights.h5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                save_weights_only=True,
                                                save_best_only= True,
                                                save_freq='epoch',
                                                verbose=1)
# Transfer learning : feater extraction

base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top = False)
base_model.trainable = False
inputs = tf.keras.layers.Input(shape=(224,224,3), name='input_layer')
x = base_model(inputs)
x = tf.keras.layers.GlobalAveragePooling2D(name='pool_layer')(x)
outputs= tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')(x)
model = tf.keras.Model(inputs,outputs)

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = 'adam',
    metrics=['accuracy']
)
initial_epoch = 4
history_1 = model.fit(train_data,
                      epochs = initial_epoch,
                      steps_per_epoch = len(train_data),
                      validation_data=test_data,
                      validation_steps = int(0.25*len(test_data)),
                      callbacks=[checkpoint])
model.save("C:/Users/walid/Desktop/Breast_Cancer_2.keras")
# Transfer learning = fine_tuning

## create baseline model
model = tf.keras.Model(inputs,outputs)
model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=["accuracy"]
)
model.load_weights(checkpoint_path)
model_base = model.layers[1]
model_base.trainable = True
# Train the 10 last layers of the model
for layer in model_base.layers[:-10]:
    layer.trainable = True

model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=["accuracy"]
)
fin_tune_epoch = initial_epoch + 2
history_final = model.fit(
    train_data,
    epochs = fin_tune_epoch,
    initial_epoch = history_1.epoch[-1],
    validation_data=test_data,
    steps_per_epoch=len(train_data),
    validation_steps=len(test_data)
)
model.save("C:/Users/walid/Desktop/Breast_Cancer_3.keras")
def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two model history objects.
    """
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    print(len(acc))

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    print(len(total_acc))
    print(total_acc)

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
compare_historys(history_1, history_final, initial_epochs=6)

