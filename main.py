import wave
import cv2
import numpy as np
import subprocess
import struct
import os
from pydub import AudioSegment

SOUND_PATH = "song" #Your sound's path without '.mp3'.
BACKGROUND_IMAGE_PATH = "./background.jpg" #Your background image's path.
FOREGROUND_IMAGE_PATH = "./logo.png" #Your logo's path.
OUTPUT_PATH = "./output/result" #Your output directory and file nime without '.mkv'.
#Increase ratio value gives more truth value but process will be slower. 32 or 64 best values for ratio.
ratio = 32 

class ImportSound():
    
    def __init__(self, sound_path) -> None:
        """
        Convert any '.mp3' sound file to '.wav' file.
        Also if sound is stereo, it will be mono.
        Args:
            sound_path (str): Your sound's path.
        """
        self.src = f"{sound_path}.mp3"
        self.dst = f"{sound_path}.wav"
        try:
            self.save_sound()
            print("[INFO] Sound converted succesfully to '.wav' file.")
        except:
            print("[ERROR] Sound couldn't converted. It must be '.mp3' file.")
    
    def save_sound(self):
        sound = AudioSegment.from_mp3(self.src)
        sound = sound.set_channels(1)
        sound.export(self.dst, format = "wav")

class GetFreqList():
    
    def __init__(self, dst) -> None:
        """
        Calculate frequencies in given sound.
        Args:
            dst (str): Your destination path.
        """
        self.dst = dst
        self.convert_to_array()
        self.get_freq_array()
        print("\n[INFO] Frequencies calculated.")
    
    def convert_to_array(self):
        """
        It converts sound to numpy array.
        """
        wav_file = wave.open(self.dst)
        self.nframes = wav_file.getnframes()
        self.frate = int(wav_file.getframerate() / ratio)
        self.data = wav_file.readframes(self.nframes)
        wav_file.close()
        self.data = struct.unpack("{n}h".format(n=self.nframes), self.data)
        self.data = np.array(self.data)
    
    def get_freq_array(self):
        """
        Calculates the frequency of the sounds in the array with FFT.
        """
        self.freq_array = []
        print("[INFO] Calculating frequencies.")
        for i in range(0, len(self.data), self.frate):
            percent = (i / len(self.data) * 100)
            print(
                f"{int(percent + 1)}% : [" + "█"*int(percent + 1) + "]", 
                end = "\r")
            
            w = np.fft.fft(self.data[i:i+self.frate])
            freqs = np.fft.fftfreq(len(w))
            idx = np.argmax(np.abs(w))
            freq = freqs[idx]
            self.freq_in_hz = int(abs(freq * self.frate * ratio))
            self.freq_array.append(self.freq_in_hz)

class CreateVideo():
    
    def __init__(self, freq_array, bg_path, fg_path, 
                glowing_effect = False, size_effect = False) -> None:
        """
        It creates image array for convert a video.
        Args:
            freq_array (list): Frequency array.
            bg_path (str): Your background image's path.
            fg_path (str): Your logo images's path.
            glowing_effect (bool, optional): If you want to glow effect. Defaults to False.
            size_effect (bool, optional): If you want to change the logo's size with sound. Defaults to False.
        """
        self.img_array = []
        self.bg_path = bg_path
        self.fg_path = fg_path
        self.freq_array = freq_array
        self.glowing_effect = glowing_effect
        self.size_effect = size_effect
        self.process()
    
    def process(self):
        """
        It creates a image array with your effects and logo.
        """
        print("[INFO] Generating video. It might be take a time.")
        for i in range(len(self.freq_array)):
            percent = (i / len(self.freq_array) * 100)
            print(
                f"{int(percent + 1)}% : [" + "█"*int(percent + 1) + "]", 
                end = "\r")
            self.background = cv2.imread(self.bg_path)
            self.foreground = cv2.imread(self.fg_path)
            self.foreground = cv2.resize(self.foreground, (300, 300))
            freq = self.freq_array[i]
            if self.glowing_effect:
                self.glowing(freq)
            if self.size_effect:
                self.size(freq)
            self.merge()
        print("\n[INFO] Video is generated.")
    
    def merge(self):
        """
        It merge two images.
        """
        width, height = self.foreground.shape[0], self.foreground.shape[1]
        x, y = int(self.background.shape[0]/2), int(self.background.shape[1]/2)
        roi = self.background[x-int(width/2):x+int(width/2), y-int(height/2):y+int(height/2)]
        foreground_gray = cv2.cvtColor(self.foreground, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(foreground_gray, 150, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        background_bg = cv2.bitwise_and(roi, roi, mask = mask)
        foreground_fg = cv2.bitwise_and(self.foreground, self.foreground, mask = mask_inv)
        
        dst = cv2.add(background_bg, foreground_fg)
        self.background[x-int(width/2):x+int(width/2), y-int(height/2):y+int(height/2)] = dst
        
        self.img_array.append(self.background)
    
    def glowing(self, freq):
        """
        Creates a glow effect where there are low frequency sounds (bass).
        Args:
            freq (uint): Frequency value.
        """
        hsv = cv2.cvtColor(self.background, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        lim = 255 - freq
        v[v <= lim] += freq
        merged = cv2.merge((h,s,v))
        self.background = cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)
        
    def size(self, freq):
        """
        It increase the size of the logo where there are low frequency sounds (bass).
        Args:
            freq (uint): Frequency value.
        """
        if freq == 0:
            freq = 1
        size = 300 + int(200/freq) * 2
        self.foreground = cv2.resize(self.foreground, (size, size))

class SaveVideo():
    
    def __init__(self, dst, img_array, isDel = False) -> None:
        """
        Saves the given image arrays in the given directory.
        Args:
            dst (str): Sound file's destination.
            img_array (list): Images array.
            isDel (bool, optional): If you want to delete temp files, you could set the value True. Defaults to False.
        """
        self.dst = dst
        self.img_array = img_array
        fourcc = cv2.VideoWriter_fourcc (*'MJPG')
        width = self.img_array[0].shape[0]
        height = self.img_array[0].shape[1]
        self.out = cv2.VideoWriter(f"{SOUND_PATH}.avi", fourcc, ratio, (height, width))
        self.savevideo()
        self.out.release()
        self.merge_sound_and_video()
        if isDel:
            self.del_empty_files()
        
    def savevideo(self):
        """
        It makes a video with given image array and save.
        """
        print("[INFO] Video is saving.")
        for i in range(len(self.img_array)):
            percent = i / len(self.img_array) * 100 
            print(f"{int(percent+1)}% : [" + "█"*int(percent+1) + "]", end="\r")
            self.out.write(self.img_array[i])

    def merge_sound_and_video(self):
        """
        It merges a sound and a video.
        """
        text = f"ffmpeg -y -i {SOUND_PATH}.wav  -r 30 -i {SOUND_PATH}.avi  -filter:a aresample=async=1 -c:a flac -c:v copy {OUTPUT_PATH}.mkv"
        cmd = text
        subprocess.call(cmd, shell=True)

    def del_empty_files(self):
        """
        It deletes temp files.
        """
        os.remove(f"{SOUND_PATH}.wav")
        os.remove(f"{SOUND_PATH}.avi")

if __name__ == "__main__":
    sound = ImportSound(SOUND_PATH)
    freqs = GetFreqList(sound.dst)
    freq_array = freqs.freq_array
    video = CreateVideo(freq_array, BACKGROUND_IMAGE_PATH, FOREGROUND_IMAGE_PATH,
                        glowing_effect=True, size_effect=True)
    SaveVideo(sound.dst, video.img_array, isDel = True)
    print(f"[INFO] Done! You can find your file in {OUTPUT_PATH}.")
    print("Press any key to exit.")
    input()
