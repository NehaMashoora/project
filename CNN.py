from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (224, 224, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))


classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

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

history=classifier.fit_generator(training_set,
                        steps_per_epoch=len(training_set),
                        epochs=20,
                        validation_data=test_set,
                        validation_steps=len(test_set))




classifier.save('trained_model_stroke.h5')
classifier.summary()

print("Training Accuracy:", history.history['accuracy'][-1])
print("Validation Accuracy:", history.history['val_accuracy'][-1])