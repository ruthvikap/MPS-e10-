import speech_recognition as sr
import numpy as np
import matplotlib.pyplot as plt
import cv2
from easygui import *
import os
from PIL import Image, ImageTk
from itertools import count
import tkinter as tk
import string

def func():
    r = sr.Recognizer()     #input
    isl_gif = ['address', 'ahemdabad', 'all', 'any questions', 'are you angry', 'are you hungry', 'assam', 'august', 
    'banana', 'banaras', 'banglore', 'be careful', 'bridge', 'cat', 'christmas', 'church', 'clinic', 'dasara', 
    'december', 'did you finish homework', 'do you have money', 'do you want something to drink', 'do you watch TV', 
    'dont worry', 'flower is beautiful', 'good afternoon', 'good morning', 'good question', 'grapes', 'hello', 
    'hindu', 'hyderabad', 'i am a clerk', 'i am fine', 'i am sorry', 'i am thinking', 'i am tired', 
    'i go to a theatre', 'i had to say something but I forgot', 'i like pink colour', 'i love to shop', 'job', 
    'july', 'june', 'karnataka', 'kerala', 'krishna', 'lets go for lunch', 'mango', 'may', 'mile', 'mumbai', 
    'nagpur', 'nice to meet you', 'open the door', 'pakistan', 'please call me later', 'police station', 
    'post office', 'pune', 'punjab', 'saturday', 'shall I help you', 'shall we go together tommorow', 'shop', 
    'sign language interpreter', 'sit down', 'stand up', 'take care', 'temple', 'there was traffic jam', 
    'thursday', 'toilet', 'tomato', 'tuesday', 'usa', 'village', 'wednesday', 'what is the problem', 
    'what is today\'s date', 'what is your father do', 'what is your name', 'whats up', 'where is the bathroom', 
    'where is the police station', 'you are wrong']

    arr = list(string.ascii_lowercase)

    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)

        while True:
            print('Say something')
            audio = r.listen(source)
            try:
                a = r.recognize_google(audio)
                print("You said: " + a.lower())

                for c in string.punctuation:
                    a = a.replace(c, "")

                if a.lower() == 'goodbye':
                    print("Oops! Time to say goodbye.")
                    break

                elif a.lower() in isl_gif:  # Check if recognized phrase matches one in the list
                    class ImageLabel(tk.Label):
                        """A label that displays images and plays them if they are GIFs."""
                        def load(self, im):
                            if isinstance(im, str):
                                im = Image.open(im)
                            self.loc = 0
                            self.frames = []
                            try:
                                for i in count(1):
                                    self.frames.append(ImageTk.PhotoImage(im.copy()))
                                    im.seek(i)
                            except EOFError:
                                pass
                            try:
                                self.delay = im.info['duration']
                            except:
                                self.delay = 100

                            if len(self.frames) == 1:
                                self.config(image=self.frames[0])
                            else:
                                self.next_frame()

                        def unload(self):
                            self.config(image=None)
                            self.frames = None

                        def next_frame(self):
                            if self.frames:
                                self.loc += 1
                                self.loc %= len(self.frames)
                                self.config(image=self.frames[self.loc])
                                self.after(self.delay, self.next_frame)

                    # Check if the GIF file exists
                    gif_path = os.path.join("D:/Downloads/ISLAT/Indian-Sign-Language-Audio-Visual-Translator/ISL_Gif", f"{a.lower()}.gif")
                    if os.path.exists(gif_path):  # Verify the GIF file exists
                        root1 = tk.Tk()
                        root1.title("Sign Language Interpreter")
                        lbl = ImageLabel(root1)
                        lbl.pack()
                        lbl.load(gif_path)  # Load the GIF
                        root1.mainloop()
                    else:
                        print(f"Gif File {gif_path} does not exist.")
                else:                                                           #breakdowns phrases into texts
                    for i in range(len(a)):
                        if a[i] in arr:
                            ImageAddress = os.path.join("D:/Downloads/ISLAT/Indian-Sign-Language-Audio-Visual-Translator/letters", f"{a[i]}.jpg")
                            if os.path.exists(ImageAddress):  # Check if the image exists
                                ImageItself = Image.open(ImageAddress)
                                ImageNumpyFormat = np.asarray(ImageItself)
                                plt.imshow(ImageNumpyFormat)
                                plt.axis('off')  # Hide axes
                                plt.draw()
                                plt.pause(0.8)  # Pause how many seconds
                            else:
                                print(f"Image file {ImageAddress} does not exist.")
                        else:
                            continue

            except Exception as e:
                print(f"Could not listen: {e}")
            plt.close()

# Main loop for starting the program
while True:
    image_path = "D:/Downloads/ISLAT/Indian-Sign-Language-Audio-Visual-Translator/mainpage.png"  # Update this path as needed

    if not os.path.exists(image_path):  # Check if the image exists before showing the GUI
        print(f"Image file {image_path} does not exist.")
        break

    msg = "INDIAN SIGN LANGUAGE AUDIO VISUAL TRANSLATOR [ISLAT]"
    choices = ["Live Voice", "All Done!"]
    reply = buttonbox(msg, image=image_path, choices=choices)

    if reply == choices[0]:
        func()
    elif reply == choices[1]:
        print("Exiting program.")
        break
print("Program terminated.")