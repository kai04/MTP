import sys
import math
from pydub import AudioSegment
from pydub.utils import make_chunks

filename = sys.argv[1]
myaudio = AudioSegment.from_file(filename , "mp3") 
duration=math.floor(myaudio.duration_seconds)
no_seg=int(math.floor(duration/15))

for seg in range(0,no_seg):
	start = (seg*15)
	end = (start+30)
	t1 =start * 1000
	t2 =end * 1000
	tempAudio = myaudio[t1:t2]
	chunk_name = filename+"_{0}.mp3".format(seg)
	tempAudio.export(chunk_name, format="mp3")
#chunk_length_ms = 1000 # pydub calculates in millisec
#chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files
 #t1 = t1 * 1000 #Works in milliseconds
 #t2 = t2 * 1000
 #newAudio = AudioSegment.from_wav("oldSong.wav")
 #newAudio = newAudio[t1:t2]
 #newAudio.export('newSong.wav', format="wav")
#for i, chunk in enumerate(chunks):
#    chunk_name = "chunk{0}.wav".format(i)
#    print "exporting", chunk_name
#    chunk.export(chunk_name, format="wav")
