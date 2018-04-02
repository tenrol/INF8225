import numpy as np
import pandas as pd
import os
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("model", help="Numéro du modèle", type=int)
parser.parse_args()
args = parser.parse_args()

MODEL = args.model

print("Modèle choisi ", MODEL)

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM, Dropout, Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.callbacks import CSVLogger

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt


# fix random seed for reproducibility
np.random.seed(7)

EMBEDDING_DIM = 50
MAX_SEQUENCE_LENGTH = 500
top_words = 20000
batch_size=256
epochs=10

GLOVE_DIR = '/usagers/lovar/'
GLOVE_FILE = 'glove.6B.' + str(EMBEDDING_DIM) + 'd.txt'
DIR_SAVED_MODEL = 'models/'
FILENAME_SAVED_MODEL = 'model_' + str(MODEL) + "_dim_" + str(EMBEDDING_DIM) + '.h5'
TRAINING_DATASET_FILE = 'clean_training140.csv'
TEST_DATASET_FILE = 'clean_test140.csv'
DIR_SCORE = "scores/"
DIR_FIG = "figures/"
FILENAME_SCORE = "score_" + str(MODEL) + ".csv"
FILENAME_FIG_ACC = "acc_" + str(MODEL) + ".png"
FILENAME_FIG_LOSS = "loss_" + str(MODEL) + ".png"
FILENAME_FIG_MODEL = "model" + str(MODEL) + ".png"


data_dict = {"text": str, "label": int}

df=pd.read_csv(TRAINING_DATASET_FILE, dtype=data_dict)
print(len(df["text"]))
df = df[df.text.notnull()]
print(len(df["text"]))
labels = df["label"]
texts = df["text"]

tokenizer = Tokenizer(top_words)
tokenizer.fit_on_texts(texts)
 
sequences = tokenizer.texts_to_sequences(texts)
 
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
 
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
 
encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
labels = np_utils.to_categorical(encoded_Y)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 42)

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, GLOVE_FILE))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
print(encoder.classes_)
print("Shape labels: " + str(y_train.shape))
print("Shape data: " + str(X_train.shape))

early_stopping = EarlyStopping(monitor='val_acc', patience=3)
csv_logger = CSVLogger(os.path.join(DIR_SCORE, FILENAME_SCORE))

# create the model
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True))
print("Choix du modèle: ", MODEL)				
if MODEL == 1:
	model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
	model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
	model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=5))
	model.add(Dropout(0.2))
	model.add(Bidirectional(LSTM(32, return_sequences=True)))
	model.add(Bidirectional(LSTM(32)))
	model.add(Dropout(0.2))
	model.add(Dense(y_train.shape[1], activation='softmax'))
	# 1 epoch = 1820s
	
elif MODEL == 2:
	model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=5))
	model.add(Dropout(0.2))
	model.add(LSTM(32))
	model.add(Dropout(0.2))
	model.add(Dense(y_train.shape[1], activation='softmax'))
	# 1 epoch = 900s

elif MODEL == 3:
	model.add(Bidirectional(LSTM(32, return_sequences=True)))
	model.add(Bidirectional(LSTM(32)))
	model.add(Dense(y_train.shape[1], activation='softmax'))

elif MODEL == 4:
	model.add(Dropout(0.25))
	model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=5))
	model.add(Dropout(0.2))
	model.add(LSTM(256))
	model.add(Dropout(0.2))
	model.add(Dense(y_train.shape[1], activation='sigmoid'))

elif MODEL == 5:
	model.add(Dropout(0.25))
	model.add(Conv1D(filters=64, kernel_size=5, padding='valid', activation='relu', strides=1))
	model.add(MaxPooling1D(pool_size=4))
	model.add(Bidirectional(LSTM(70)))
	model.add(Dense(y_train.shape[1], activation='sigmoid'))

else:
	print("Ce modèle n'existe pas: ", MODEL)
	exit()

plot_model(model, to_file=os.path.join(DIR_FIG, FILENAME_FIG_MODEL), show_shapes=True)
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

print(model.summary())

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2
, callbacks=[early_stopping, csv_logger])
model.save(os.path.join(DIR_SAVED_MODEL, FILENAME_SAVED_MODEL))

# Final evaluation of the model
score = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (score[1]*100))

# summarize history for accuracy
fig_acc = plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
fig_acc.savefig(os.path.join(DIR_FIG, FILENAME_FIG_ACC))

# summarize history for loss
fig_loss = plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
fig_loss.savefig(os.path.join(DIR_FIG, FILENAME_FIG_LOSS))


# Ecriture dans le fichier des resultats du test
fout = open(os.path.join(DIR_SCORE, FILENAME_SCORE), 'a')
fout.write("test loss,test accuracy\n\n")

fout.write(str(score[0]) + "," + str(score[1]) + "\n")
fout.close()

model.save(os.path.join(DIR_SAVED_MODEL, FILENAME_SAVED_MODEL))
