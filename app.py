import streamlit as st
import numpy as np
import pickle

# Load the trained model
loaded_model = pickle.load(open("trained_model.sav", 'rb'))
scaler = pickle.load(open("scaler.sav", 'rb'))

#creating a function for prediction
def audio_prediction(input_data):
    
    #changing the input data to numpy array
    input_data_as_np_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_np_array.reshape(1, -1)

    input_data_standardized = scaler.transform(input_data_reshaped)

    prediction = loaded_model.predict(input_data_standardized)
    print(prediction)

    if (prediction[0]==1):
        return "This person is real"
    else:
        return "This person is fake"


def main():
    # Title of the web app
    st.title("Real vs Fake Audio Classifier")

    # Instructions
    st.write("Provide the following audio features to predict whether the audio is real or fake:")

    # Input fields for audio features
    frequency = st.text_input("Frequency")
    speech_rate = st.text_input("Speech Rate")
    pitch_variability = st.text_input("Pitch Variability")
    zcr = st.text_input("Zero Crossing Rate (ZCR)")
    rms_energy = st.text_input("RMS Energy")
    temporal_centroid = st.text_input("Temporal Centroid")
    spectral_centroid = st.text_input("Spectral Centroid")
    spectral_bandwidth = st.text_input("Spectral Bandwidth")
    spectral_contrast = st.text_input("Spectral Contrast")
    spectral_flatness = st.text_input("Spectral Flatness")
    mfccs = st.text_input("MFCCs")
    chroma_vector = st.text_input("Chroma Vector")
    hnr = st.text_input("Harmonic-to-Noise Ratio (HNR)")
    tonal_centroid = st.text_input("Tonal Centroid")

    # code for Prediction
    prediction = ''
    
    # creating a button for Prediction
    if st.button('Classify Audio'):
        prediction = audio_prediction([frequency, speech_rate, pitch_variability, zcr, rms_energy, temporal_centroid, spectral_centroid, spectral_bandwidth,
                                        spectral_contrast, spectral_flatness, mfccs, chroma_vector, hnr, tonal_centroid])
        
    st.success(prediction)

if __name__ == '__main__':
    main()