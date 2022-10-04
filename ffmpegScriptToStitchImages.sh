#!/bin/sh

#!/bin/bash

FOLDER="Sorted"
FILES=$FOLDER/*

mkdir $FOLDER/Videos

for f in $FILES
do
  echo "Processing $f"
  cd $f
  rename 's/\d+/sprintf("%08d", $&)/e' *.png
  VIDNAME=$(echo $f | cut -d "/" -f 2)
  echo $VIDNAME

  # take action on each file. $f store current file name
  ffmpeg -framerate 30 -pattern_type glob -i "*.png" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -crf 20 -pix_fmt yuv420p ../Videos/$VIDNAME.mp4 -y
  cd ../..

done
