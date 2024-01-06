# SoundSec-CNN-LSTM

## Overview
`SoundSec-CNN-LSTM` is an advanced project that employs Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks for recognizing security-sensitive sounds. The project is particularly focused on analyzing mel-spectrogram images, which provide a robust representation for audio processing tasks.

## Project Description
This project aims to develop a reliable and efficient system for the identification of sounds associated with security threats. By integrating CNNs and LSTMs, it captures spatial features within mel-spectrograms and temporal patterns in audio sequences, making it adept at recognizing a variety of security-related samples.

## Detailed Methodology
For an in-depth explanation of the methodologies and theoretical frameworks employed in this project, please refer to the following document:
- [SoundSec-CNN-LSTM Methodology (PDF)](SoundSec-CNN-LSTM.pdf)

## Key Features
- **CNN-LSTM Architecture:** Combines CNNs for extracting features from mel-spectrogram images with LSTMs for understanding temporal dependencies.
- **Mel-Spectrogram Analysis:** Utilizes mel-spectrograms for a frequency-based, detailed representation of audio data.
- **Security Sound Recognition:** Specializes in detecting and classifying sounds that could indicate security threats, such as alarms, breaking glass, dog barks, explosions, sirens, screaming and gunshots.

## Dataset
The dataset comprises 2524 security-suspicious sounds, averaging about 5 seconds in length and categorized into 7 classes. The classes include alarms, dog barking, explosions, glass breaking, screaming, shooting, and sirens. This diverse collection enables the model to learn and recognize a wide range of security-related acoustic signatures.

## Acknowledgments

This project is based on and includes modifications to code originally developed by Hbbbbbby in their EmotionRecognition_2Dcnn-lstm project. The original code is licensed under the BSD 3-Clause License. We extend our gratitude to Hbbbbbby for their contributions to the open-source community. The original project can be found [here](https://github.com/Hbbbbbby/EmotionRecognition_2Dcnn-lstm).

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
