#!/bin/bash 
#tmp=${a#*_}   # remove prefix ending in "_"
#b=${tmp%_*}   # remove suffix starting with "_"
#b=${a:12:5}   #where 12 is the offset (zero-based) and 5 is the length
ls -d *.mp3 >tmp.txt
dir_name="pitches"
mkdir $dir_name
dir_name="./"$dir_name"/	"
cat "tmp.txt" | while read LINE
do
	chmod a+x $LINE
	fname=$dir_name$LINE 
	#tmp=${LINE%mp3}
	#echo "$tmp"
	echo "$fname"
	python2 pitch_example.py "$fname" 
	#sox $LINE {$LINE}.wav
done
rm "tmp.txt"
