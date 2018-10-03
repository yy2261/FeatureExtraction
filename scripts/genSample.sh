#!/bin/bash

#echo "input project name:"
proj=$1

includePath='/media/yy/10A4078410A40784/grad_proj/data/word2vecDicts/withMethodVec.txt'

python /media/yy/10A4078410A40784/grad_proj/code/genSamples.py \
/media/yy/10A4078410A40784/grad_proj/exp/stem_feature/${proj}/ \
${includePath} \
/media/yy/10A4078410A40784/grad_proj/exp/dicts/${proj}_with.npy

notIncludePath='/media/yy/10A4078410A40784/grad_proj/data/word2vecDicts/withoutMethodVec.txt'

python /media/yy/10A4078410A40784/grad_proj/code/genSamples.py \
/media/yy/10A4078410A40784/grad_proj/exp/stem_feature/${proj}/ \
${includePath} \
/media/yy/10A4078410A40784/grad_proj/exp/dicts/${proj}_without.npy