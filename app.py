from flask import Flask, render_template, redirect, request
from keras.models import load_model
from scipy.io import wavfile
import numpy as np 
import librosa
from scipy.stats import zscore


app = Flask(__name__) 

@app.route("/", methods=["POST","GET"])
def index():
    transcript = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            audioFile,sr = librosa.core.load(file,sr=21500,offset=0.5) #input audio file
            audioFile = zscore(audioFile)   #normalize

            #PREPROCESSING
            max_pad_len = 65000   
            if len(audioFile)<max_pad_len:   #pad or truncate to 3 secs
                audioFile_padded = np.zeros(max_pad_len)
                audioFile_padded[:len(audioFile)] = audioFile
                audioFile = audioFile_padded
            elif len(audioFile)> max_pad_len:
                audioFile = np.asarray(audioFile[:max_pad_len])

            #FEATURE EXTRACTION
            mel_spect = np.abs(librosa.stft(audioFile, n_fft=512, window='hamming', win_length=256, hop_length=128)) ** 2
    
            # Compute mel spectrogram
            mel_spect = librosa.feature.melspectrogram(S=mel_spect, sr=21500, n_mels=128, fmax=4000)
            
            # Compute log-mel spectrogram
            mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

            print(mel_spect.shape)

            mel_spect = mel_spect.reshape(1,128,508)
            print(mel_spect.shape)

            win_ts = 128
            hop_ts = 64

            def frame(x, win_step=128, win_size=64):
                nb_frames = 1 + int((x.shape[2] - win_size) / win_step)
                frames = np.zeros((x.shape[0], nb_frames, x.shape[1], win_size)).astype(np.float32)
                for t in range(nb_frames):
                    frames[:,t,:,:] = np.copy(x[:,:,(t * win_step):(t * win_step + win_size)]).astype(np.float32)
                return frames
            mel_spect = frame(mel_spect, hop_ts, win_ts)

            mel_spect = mel_spect.reshape(mel_spect.shape[0],mel_spect.shape[1],mel_spect.shape[2],mel_spect.shape[3],1)
            model = load_model('[CNN-LSTM]Model_with_aug.hdf5')
            prediction = model.predict(mel_spect)

            label_map = ['Neutral','Happy','Sad','Angry','Fear','Disgust','Surprise']

            predict = np.argmax(prediction)
            result = label_map[predict]
            transcript = result

    return render_template('index.html', transcript=transcript)

if __name__ == "__main__":
  app.run(debug=True, threaded=True)
