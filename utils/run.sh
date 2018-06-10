ls -d *.mp3 >tmp
cat "tmp" | while read LINE
do
	chmod a+x $LINE
	#tail -n +2 "$LINE" > tmp1 && mv tmp1 $LINE
	#gawk -f scr.awk $LINE > tmp > out && mv out $LINE
	python2 pitch_example.py $LINE	
	#rm tmp
	#rm tmp1  
	#tmp=${LINE%mp3}
	#echo "${tmp}wav"
	#sox $LINE {$LINE}.wav
done
rm tmp
