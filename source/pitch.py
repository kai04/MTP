import sys, csv
import wave
import contextlib
sys.path.append("/usr/local/lib/python2.7/site-packages")
from essentia import *
from essentia.standard import *
from pylab import *
from numpy import *
#from mutagen.mp3 import MP3

#def getDur(fname):
#    audio = MP3(fname)
#    return (audio.info.length)

#def getDuration(fname):
#    with contextlib.closing(wave.open(fname,'r')) as f:
#        frames = f.getnframes()
#        rate = f.getframerate()
#        duration = frames / float(rate)
#    return duration
#    print(duration)

try:
    filename = sys.argv[1]
except:
    print("usage: %s <input-audiofile>" % sys.argv[0])
    sys.exit()



# In this example we will extract predominant melody given an audio file by
# running a chain of algorithms.
 
# First, create our algorithms:
hopSize = 128
frameSize = 2048
sampleRate = 44100
guessUnvoiced = True

run_windowing = Windowing(type='hann', zeroPadding=3*frameSize) # Hann window with x4 zero padding
run_spectrum = Spectrum(size=frameSize * 4)
run_spectral_peaks = SpectralPeaks(minFrequency=1,
                                   maxFrequency=20000,
                                   maxPeaks=100,
                                   sampleRate=sampleRate,
                                   magnitudeThreshold=0,
                                   orderBy="magnitude")
run_pitch_salience_function = PitchSalienceFunction()
run_pitch_salience_function_peaks = PitchSalienceFunctionPeaks()
run_pitch_contours = PitchContours(hopSize=hopSize)
run_pitch_contours_melody = PitchContoursMelody(guessUnvoiced=guessUnvoiced,
                                                hopSize=hopSize)

# ... and create a Pool
pool = Pool();

# Now we are ready to start processing.
# 1. Load audio and pass it through the equal-loudness filter
audio = MonoLoader(filename=filename)()
audio = EqualLoudness()(audio)

# 2. Cut audio into frames and compute for each frame:
#    spectrum -> spectral peaks -> pitch salience function -> pitch salience function peaks
list_start_time=[]
list_duration=[]

for frame in FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
    frame = run_windowing(frame)
    spectrum = run_spectrum(frame)
    peak_frequencies, peak_magnitudes = run_spectral_peaks(spectrum)
    
    salience = run_pitch_salience_function(peak_frequencies, peak_magnitudes)
    salience_peaks_bins, salience_peaks_saliences = run_pitch_salience_function_peaks(salience)
    
    pool.add('allframes_salience_peaks_bins', salience_peaks_bins)
    pool.add('allframes_salience_peaks_saliences', salience_peaks_saliences)

# 3. Now, as we have gathered the required per-frame data, we can feed it to the contour 
#    tracking and melody detection algorithms:
contours_bins, contours_saliences, contours_start_times, duration = run_pitch_contours(
        pool['allframes_salience_peaks_bins'],
        pool['allframes_salience_peaks_saliences'])
        
pitch, confidence = run_pitch_contours_melody(contours_bins,contours_saliences,contours_start_times,duration)
print("start:",contours_start_times)
print("duration:",duration)

# NOTE that we can avoid the majority of intermediate steps by using a composite algorithm
#      PredominantMelody (see extractor_predominant_melody.py). This script will be usefull 
#      if you want to get access to pitch salience function and pitch contours.

n_frames = len(pitch)
print("number of frames: %d" % n_frames)

len_dur=float(duration/n_frames)

last_frame=0
for x in range(n_frames):
    list_start_time.append(last_frame)
    last_frame+=len_dur
    

n_frames1 = len(contours_start_times)
print("number of start: %d" % n_frames1)

wrt_fname=filename+"_pitch.csv"
f=open(wrt_fname,"w")
for x in range(n_frames):
    f.write(str(list_start_time[x])+","+str(pitch[x])+"\n")

# visualize output pitch
#==============================================================================
# fig = plt.figure()
# plot(range(n_frames), pitch, 'b')
# n_ticks = 10
# xtick_locs = [i * (n_frames / 10.0) for i in range(n_ticks)]
# xtick_lbls = [i * (n_frames / 10.0) * hopSize / sampleRate for i in range(n_ticks)]
# xtick_lbls = ["%.2f" % round(x,2) for x in xtick_lbls]
# plt.xticks(xtick_locs, xtick_lbls)
# ax = fig.add_subplot(111)
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Pitch (Hz)')
# suptitle("Predominant melody pitch")
# 
#==============================================================================


# visualize output pitch confidence
#==============================================================================
# fig = plt.figure()
# plot(range(n_frames), confidence, 'b')
# n_ticks = 10
# xtick_locs = [i * (n_frames / 10.0) for i in range(n_ticks)]
# xtick_lbls = [i * (n_frames / 10.0) * hopSize / sampleRate for i in range(n_ticks)]
# xtick_lbls = ["%.2f" % round(x,2) for x in xtick_lbls]
# plt.xticks(xtick_locs, xtick_lbls)
# ax = fig.add_subplot(111)
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Confidence')
# suptitle("Predominant melody pitch confidence")
#==============================================================================

show()
