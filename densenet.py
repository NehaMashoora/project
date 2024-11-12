from keras.applications import DenseNet121
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

model = Sequential()
model.add(base_model)

model.add(Flatten())

model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(r"C:\Users\abhia\OneDrive\Desktop\nanddg\Brain-Stroke-Prediction\dataset\train",
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory(r"C:\Users\abhia\OneDrive\Desktop\nanddg\Brain-Stroke-Prediction\dataset\test",
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='binary')

history=model.fit_generator(training_set,
                    steps_per_epoch=8000 // 32,  
                    epochs=5,
                    validation_data=test_set,
                    validation_steps=2000 // 32)  

model.save('trained_model_densenet.h5')

model.summary()
print("Training Accuracy:", history.history['accuracy'][-1])
print("Validation Accuracy:", history.history['val_accuracy'][-1])