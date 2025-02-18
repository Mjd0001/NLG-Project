import streamlit as st
import numpy as np
import pickle
from tensorflow import keras

#Load the LSTM Model
model=keras.models.load_model('next_word_lstm.h5')

#3 Laod the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)


# Function to predict the next word
max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape

def predict_next_word(text):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# streamlit app
st.title("Next Word Prediction")
input_text= st.text_input("Enter the sequence of Words","To be or not to be")

if st.button("Predict Next Word"):
    next_word = predict_next_word(input_text)
    st.write(f'Next word: {next_word}')

