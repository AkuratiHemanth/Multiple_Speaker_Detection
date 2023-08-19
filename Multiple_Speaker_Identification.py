''' Importing essential libraries such as TensorFlow for machine learning, file handling tools like os and shutil, 
numerical computations with numpy, high-level neural networks API keras, path manipulation with pathlib, 
content display via IPython.display, and system command execution via subprocess. 📚💻🔧'''

import tensorflow as tf
import os
from os.path import isfile, join
import numpy as np
import shutil
from tensorflow import keras
from pathlib import Path
from IPython.display import display, Audio
import subprocess


'''Copying the "speaker-recognition-dataset" directory 
from the input location to the current directory using the command !cp -r "../input/speaker-recognition-dataset" ./. 📂🔁'''

!cp -r "../input/speaker-recognition-dataset" ./

'''Defining paths for the dataset: data_directory points to the "./speaker-recognition-dataset/16000_pcm_speeches" directory, 
while audio_folder and noise_folder store the subfolders "audio" and "noise," respectively. 
The complete paths are created using os.path.join(). 📂🔊📁'''

data_directory = "./speaker-recognition-dataset/16000_pcm_speeches"
audio_folder = "audio"
noise_folder = "noise"

audio_path = os.path.join(data_directory, audio_folder)
noise_path = os.path.join(data_directory, noise_folder)


'''Setting hyperparameters for the model:

valid_split: Validation data split ratio is 10%.
shuffle_seed: Seed value for shuffling data is 43.
sample_rate: Audio sample rate is 16000 Hz.
scale: Scaling factor is 0.5.
batch_size: Training batch size is 128.
epochs: Training for 15 epochs. 🎚️🧮🔢'''

valid_split = 0.1

shuffle_seed = 43

sample_rate = 16000

scale = 0.5

batch_size = 128

epochs = 15

'''Organizing the dataset folders:

For each folder in data_directory:
If the folder is a subdirectory:
If it's either "audio" or "noise," skip.
Else if it's "other" or "background_noise":
Move the folder to noise_path.
Otherwise:
Move the folder to audio_path.
This code snippet is responsible for categorizing and moving folders within the dataset structure for better organization. 📂🗂️📁'''


for folder in os.listdir(data_directory):
    if os.path.isdir(os.path.join(data_directory, folder)):
        if folder in [audio_folder, noise_folder]:
            
            continue
        elif folder in ["other", "_background_noise_"]:
            
            shutil.move(
                os.path.join(data_directory, folder),
                os.path.join(noise_path, folder),
            )
        else:
            shutil.move(
                os.path.join(data_directory, folder),
                os.path.join(audio_path, folder),
            )




'''Creating a list noise_paths to store paths of noise audio files:

For each subdir in noise_path:
subdir_path is formed using Path(noise_path) / subdir.
If subdir_path is a directory:
Iterate through files within it:
If the file has ".wav" extension, add its path to noise_paths.
This code gathers paths of noise audio files to be used in the project. 🔉📁🎚️'''


noise_paths = []
for subdir in os.listdir(noise_path):
    subdir_path = Path(noise_path) / subdir
    if os.path.isdir(subdir_path):
        noise_paths += [
            os.path.join(subdir_path, filepath)
            for filepath in os.listdir(subdir_path)
            if filepath.endswith(".wav")
        ]

'''The command variable stores a multi-line shell command for audio file processing:

For each directory dir in noise_path:
For each .wav file file in the directory:
Extract the sample_rate using ffprobe.
If sample_rate is not 16000 Hz:
Use ffmpeg to resample the audio to 16000 Hz.
Rename the resampled file to the original.
This command is designed to ensure all noise audio files are at the desired 16000 Hz sample rate. 🎧🔊🔧'''


command = (
    "for dir in `ls -1 " + noise_path + "`; do "
    "for file in `ls -1 " + noise_path + "/$dir/*.wav`; do "
    "sample_rate=`ffprobe -hide_banner -loglevel panic -show_streams "
    "$file | grep sample_rate | cut -f2 -d=`; "
    "if [ $sample_rate -ne 16000 ]; then "
    "ffmpeg -hide_banner -loglevel panic -y "
    "-i $file -ar 16000 temp.wav; "
    "mv temp.wav $file; "
    "fi; done; done"
)

'''Executing the command to resample noise audio files.

Then, defining a function load_noise_sample(path) to load noise samples:

Decode the WAV audio using tf.audio.decode_wav().
If the sampling rate matches sample_rate:
Split the audio into chunks of sample_rate length.
If sampling rate doesn't match, print a message and return None.
Next, create a list noises to store the resampled noise samples:

For each path in noise_paths:
Load the noise sample using the defined function.
If a valid sample, extend the noises list with the sample chunks.
Finally, stack the noises list using tf.stack() to create a tensor of resampled noise samples. 🎧🔊🔧🔢'''

os.system(command)
def load_noise_sample(path):
    sample, sampling_rate = tf.audio.decode_wav(
        tf.io.read_file(path), desired_channels=1
    )
    if sampling_rate == sample_rate:
        slices = int(sample.shape[0] / sample_rate)
        sample = tf.split(sample[: slices * sample_rate], slices)
        return sample
    else:
        print("Sampling rate for",path, "is incorrect")
        return None


noises = []
for path in noise_paths:
    sample = load_noise_sample(path)
    if sample:
        noises.extend(sample)
noises = tf.stack(noises)

'''Creating a function paths_and_labels_to_dataset(audio_paths, labels) to convert audio paths and labels into a dataset:

Create a dataset path_ds from audio paths using tf.data.Dataset.from_tensor_slices().
Map the paths to audio using the function path_to_audio(x).
Create a label dataset label_ds from labels using tf.data.Dataset.from_tensor_slices().
Zip the audio dataset and label dataset using tf.data.Dataset.zip() and return the resulting dataset.
This function allows you to efficiently create a dataset for audio paths and their corresponding labels. 🎧🔢📂'''

def paths_and_labels_to_dataset(audio_paths, labels):
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: path_to_audio(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


'''Defining the function path_to_audio(path) to read and decode audio from a given path:

Read the audio file using tf.io.read_file().
Decode the audio using tf.audio.decode_wav(), specifying the number of channels as 1 and the sample rate.
This function facilitates reading and decoding audio from file paths, preparing them for further processing. 🔊📂🔢'''


def path_to_audio(path):
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, sample_rate)
    return audio

'''Defining the function add_noise(audio, noises=None, scale=0.5) to incorporate noise into audio samples:

If noises are provided:
Generate random indices using tf.random.uniform() to select noise samples.
Gather selected noise samples using tf.gather().
Calculate proportional factors to match audio amplitudes with noise.
Add scaled noise to audio samples.
This function allows you to blend noise with audio samples, enhancing their realism or diversity. 🔊🔉🔀🔧'''

def add_noise(audio, noises=None, scale=0.5):
    if noises is not None:
        tf_rnd = tf.random.uniform(
            (tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
        )
        noise = tf.gather(noises, tf_rnd, axis=0)

        prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
        prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)

        audio = audio + noise * prop * scale

    return audio

'''Creating the function audio_to_fft(audio) to convert audio to its Fast Fourier Transform (FFT) representation:

Squeeze the audio tensor to remove any singleton dimensions.
Compute the FFT using tf.signal.fft(), casting audio to complex numbers.
Expand dimensions to match the original shape.
Return the magnitude of the FFT, considering only the first half of the frequency bins.
This function aids in transforming audio data into its frequency domain representation for analysis. 🎵🔀🔍'''

def audio_to_fft(audio):
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])


'''Creating a dataset from audio paths and their corresponding labels:

class_names contains the list of directory names in audio_path.
Loop through each name (speaker name) in class_names:
Print the speaker's name.
Form the dir_path using Path(audio_path) / name.
Gather the paths of .wav files within the speaker's directory.
Extend audio_paths with speaker sample paths and labels with corresponding labels.
This loop constructs a list of audio paths and labels for dataset creation. 🎤📁🔢'''


class_names = os.listdir(audio_path)
print(class_names,)

audio_paths = []
labels = []
for label, name in enumerate(class_names):
    print("Speaker:",(name))
    dir_path = Path(audio_path) / name
    speaker_sample_paths = [
        os.path.join(dir_path, filepath)
        for filepath in os.listdir(dir_path)
        if filepath.endswith(".wav")
    ]
    audio_paths += speaker_sample_paths
    labels += [label] * len(speaker_sample_paths)

  
'''Shuffling the audio_paths and labels arrays to generate random data for training and validation. 
This ensures diversity and avoids any inherent order bias during model training. 🔀🔢📊'''


rng = np.random.RandomState(shuffle_seed)
rng.shuffle(audio_paths)
rng = np.random.RandomState(shuffle_seed)
rng.shuffle(labels)


'''Dividing the data into training and validation sets:

num_val_samples is calculated based on the validation split ratio.
train_audio_paths and train_labels store the paths and labels for the training set.
valid_audio_paths and valid_labels hold the paths and labels for the validation set.
This separation ensures distinct datasets for training and validating the model. 📊📚🧑‍🎓'''


num_val_samples = int(valid_split * len(audio_paths))
train_audio_paths = audio_paths[:-num_val_samples]
train_labels = labels[:-num_val_samples]


valid_audio_paths = audio_paths[-num_val_samples:]
valid_labels = labels[-num_val_samples:]



'''Creating two datasets:

train_ds: For training, paths and labels are converted into a dataset. It's shuffled and batched.
valid_ds: For validation, a similar process is applied, but with smaller buffer and batch sizes.
These datasets facilitate efficient training and validation of the model. 📊🧑‍🎓🧪'''

train_ds = paths_and_labels_to_dataset(train_audio_paths, train_labels)
train_ds = train_ds.shuffle(buffer_size=batch_size * 8, seed=shuffle_seed).batch(
    batch_size
)

valid_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=shuffle_seed).batch(32)



'''Enhancing training set:

Adding noise using add_noise() and modifying audio samples.
Transforming audio to frequency domain using audio_to_fft().
Applying similar frequency domain transformation to validation set.

Using num_parallel_calls=tf.data.experimental.AUTOTUNE maximizes parallelism and efficiency.

Prefetching further optimizes dataset pipeline performance.

These preprocessing steps prepare data for model training and validation. 🔊🔀🔍🚀'''



train_ds = train_ds.map(
    lambda x, y: (add_noise(x, noises, scale=scale), y),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
)


train_ds = train_ds.map(
    lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
)

train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

valid_ds = valid_ds.map(
    lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
)
valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)



'''Defining a Residual Network model architecture:

residual_block(x, filters, conv_num): Building a residual block with convolutions, skip connections, and activations.
build_model(input_shape, num_classes): Constructing the overall model architecture with stacked residual blocks and fully connected layers.
Compiling the model:

Using "Adam" optimizer and "sparse_categorical_crossentropy" loss.
Monitoring accuracy as a metric.
Adding callbacks:

earlystopping_cb: Early stopping with patience and restoration of best weights.
mdlcheckpoint_cb: Model checkpointing based on validation accuracy.
This code establishes, compiles, and configures the model for training and evaluation. 🏗️🧠📊'''


from tensorflow.keras.layers import Conv1D
def residual_block(x, filters, conv_num = 3, activation = "relu"):
    s = keras.layers.Conv1D(filters, 1, padding = "same")(x)
    
    for i in range(conv_num - 1):
        x = keras.layers.Conv1D(filters, 3, padding = "same")(x)
        x = keras.layers.Activation(activation)(x)
    
    x = keras.layers.Conv1D(filters, 3, padding = "same")(x)
    x = keras.layers.Add()([x, s])
    x = keras.layers.Activation(activation)(x)
    
    return keras.layers.MaxPool1D(pool_size = 2, strides = 2)(x)

def build_model(input_shape, num_classes):
    inputs = keras.layers.Input(shape = input_shape, name = "input")
    
    x = residual_block(inputs, 16, 2)
    x = residual_block(inputs, 32, 2)
    x = residual_block(inputs, 64, 3)
    x = residual_block(inputs, 128, 3)
    x = residual_block(inputs, 128, 3)
    x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    
    outputs = keras.layers.Dense(num_classes, activation = "softmax", name = "output")(x)
    
    return keras.models.Model(inputs = inputs, outputs = outputs)

model = build_model((sample_rate // 2, 1), len(class_names))

model.summary()

model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]) 

model_save_filename = "model.h5"

earlystopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(model_save_filename, monitor="val_accuracy", save_best_only=True)


'''Training the model:

Using the train_ds dataset.
Training for the specified number of epochs.
Validating on the valid_ds dataset.
Utilizing earlystopping_cb and mdlcheckpoint_cb callbacks.
This step triggers the actual training process, monitoring performance and saving checkpoints. 🏋️‍♂️🚀📈'''


history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=valid_ds,
    callbacks=[earlystopping_cb, mdlcheckpoint_cb],
)

'''
Evaluating Model Accuracy:

Printing the accuracy of the trained model by evaluating it on the valid_ds dataset. 
This provides an insight into the model's performance on unseen validation data. 📊🧪📈'''

print("Accuracy of model:",model.evaluate(valid_ds))


'''Testing and Displaying Results:

Create a test dataset similar to training/validation sets.
Iterate through a sample batch:
Transform audios and predict labels using the trained model.
Randomly select samples and compare predicted and true labels.
Print the speaker's name, prediction, and a welcome or sorry message based on correctness.
This code section assesses the model's performance and provides an interactive display of predictions. 🎤🔮📊'''


SAMPLES_TO_DISPLAY = 10

test_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
test_ds = test_ds.shuffle(buffer_size=batch_size * 8, seed=shuffle_seed).batch(
    batch_size
)

test_ds = test_ds.map(lambda x, y: (add_noise(x, noises, scale=scale), y))

for audios, labels in test_ds.take(1):
    ffts = audio_to_fft(audios)
    y_pred = model.predict(ffts)
    rnd = np.random.randint(0, batch_size, SAMPLES_TO_DISPLAY)
    audios = audios.numpy()[rnd, :, :]
    labels = labels.numpy()[rnd]
    y_pred = np.argmax(y_pred, axis=-1)[rnd]

    for index in range(SAMPLES_TO_DISPLAY):
        print(
            "Speaker:\33{} {}\33[0m\tPredicted:\33{} {}\33[0m".format(
                "[92m" if labels[index] == y_pred[index] else "[91m",
                class_names[labels[index]],
                "[92m" if labels[index] == y_pred[index] else "[91m",
                class_names[y_pred[index]],
            )
        )
        if labels[index] ==y_pred[index]:
            print("Welcome")
        else:
            print("Sorry")
        print("The speaker is" if labels[index] == y_pred[index] else "", class_names[y_pred[index]])


'''Creating a function to predict a single audio sample:

paths_to_dataset(audio_paths): Converts audio paths to a dataset.
predict(path, labels): Predicts using the model on given paths and labels:
Shuffle, batch, and prefetch the test dataset.
Apply noise and transformation operations.
Predict and display the result.
This function allows for individual audio sample prediction and display. 🔮🔊📊'''


 def paths_to_dataset(audio_paths):
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    return tf.data.Dataset.zip((path_ds))

def predict(path, labels):
    test = paths_and_labels_to_dataset(path, labels)


    test = test.shuffle(buffer_size=batch_size * 8, seed=shuffle_seed).batch(
    batch_size
    )
    test = test.prefetch(tf.data.experimental.AUTOTUNE)


    test = test.map(lambda x, y: (add_noise(x, noises, scale=scale), y))

    for audios, labels in test.take(1):
        ffts = audio_to_fft(audios)
        y_pred = model.predict(ffts)
        rnd = np.random.randint(0, 1, 1)
        audios = audios.numpy()[rnd, :]
        labels = labels.numpy()[rnd]
        y_pred = np.argmax(y_pred, axis=-1)[rnd]

    for index in range(1):
            print(
            "Speaker:\33{} {}\33[0m\tPredicted:\33{} {}\33[0m".format(
            "[92m",y_pred[index],
                "[92m", y_pred[index]
                )
            )
            
            print("Speaker Predicted:",class_names[y_pred[index]])


'''Testing the prediction function:

Provide a single audio path and an "unknown" label.
Try predicting using the given path and labels.
If an error occurs, print an error message to indicate the issue.
This code helps identify potential errors in the prediction process. 🎤🔮🔍'''


path = ["../input/speaker-recognition-dataset/16000_pcm_speeches/Jens_Stoltenberg/1013.wav"]
labels = ["unknown"]
try:
    predict(path, labels)
except:
    print("Error! Check if the file correctly passed or not!")

