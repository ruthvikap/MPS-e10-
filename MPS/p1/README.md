# Indian Sign Language Audio-to-Visual Translator (ISLAT)

**An application designed to convert spoken language into Indian Sign Language (ISL) visuals, enabling effective communication with the deaf and speech-impaired community.**

---

## Features
- **User-Friendly Interface**: Built using EasyGui for simplicity and ease of use.
- **Speech Recognition**: Real-time transcription through Google Speech API for online use and CMU Sphinx for offline functionality.
- **Text Preprocessing**: Employs natural language processing for accurate text handling and ISL mapping.
- **ISL Visualization**: Displays ISL gestures using animated GIFs for phrases and static images for individual letters.
- **Scalability**: A modular design that allows for the inclusion of a more extensive ISL vocabulary and non-manual signs in the future.

---

## Objectives
The **Indian Sign Language Audio-to-Visual Translator (ISLAT)** aims to:
1. Bridge the communication gap between the deaf and hearing communities.
2. Provide real-time, accurate translation of spoken language into ISL visuals.
3. Create a scalable and user-friendly tool for various societal contexts like education, public spaces, and healthcare.

---

## Algorithm Overview

1. **Speech Input**:
   - Captures real-time audio using PyAudio.
   - Calibrates for ambient noise for better accuracy.
2. **Speech Recognition**:
   - Converts audio to text using Sphinx (offline) or Google Speech API (online).
3. **Text Processing**:
   - Normalizes text (lowercasing, removing punctuation).
   - Matches phrases or decomposes text into letters.
4. **ISL Mapping**:
   - Retrieves ISL visuals (GIFs for phrases, images for letters) using a dictionary-based approach.
5. **Visualization**:
   - Displays ISL outputs via Tkinter (for GIFs) or Matplotlib (for letter images).
6. **Error Handling**:
   - Manages unrecognized input gracefully.
7. **User Feedback**:
   - Displays intuitive outputs and error messages for better usability.

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Required libraries: `speech_recognition`, `pyaudio`, `Pillow`, `numpy`, `matplotlib`, `easygui`, `tkinter`.

### Steps to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ISLAT.git
   cd ISLAT
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the application:
   ```bash
   python main.py

### Usage
1. **Launch the application and choose Live Voice from the menu**.
2. **Speak into the microphone. The system will:**
   - Recognize speech.
   - Convert it to text.
   - Display corresponding ISL visuals or GIFs.
3. **To exit**, say **goodbye** or select the exit option.

---

### Results

#### Example Outputs
- **Phrase Recognition**:
  - **Input**: "Good morning."
  - **Output**: Displays the GIF for "Good morning" in ISL.

- **Character Mapping**:
  - **Input**: "hello."
  - **Output**: Sequential display of hand gestures for each letter in "hello."

---

### Future Scope
- Incorporation of non-manual ISL components (facial expressions, head movements).
- Support for multiple languages and accents.
- Development of mobile and AR/VR applications for immersive learning.
- Integration with IoT devices for real-time public service applications.

---

### References
1. **Neural Sign Language Translation**: [IEEE Xplore](https://ieeexplore.ieee.org/document/8578910).
2. **CISLR: Corpus for Indian Sign Language Recognition**: [EMNLP](https://aclanthology.org/2022.emnlp-main.707/).
3. **Indian Sign Language Recognition System using SURF with SVM and CNN**: [DOI](https://doi.org/10.1016/j.array.2022.100141).
