#!/bin/bash

echo "input project name:"
read proj

mkdir ${proj}

python /media/yy/10A4078410A40784/grad_proj/code/genStemFeature.py \
/media/yy/10A4078410A40784/grad_proj/exp/select_feature/${proj}/ \
/media/yy/10A4078410A40784/grad_proj/exp/stem_feature/${proj}/