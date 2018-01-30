#!/bin/bash

echo "input project name:"
read ProjName
echo "input version 1:"
read Version_1
echo "input version 2:"
read Version_2

cd /media/yy/10A4078410A40784/grad_proj/new_pure_feature/
mkdir ${ProjName}
mkdir new_${ProjName}

cd ${ProjName}_${Version_1}
mv * ../${ProjName}
cd ..
cd ${ProjName}_${Version_2}
mv * ../${ProjName}
cd ..

echo "removing extra tokens..."
python /media/yy/10A4078410A40784/grad_proj/code/removeToken.py /media/yy/10A4078410A40784/grad_proj/new_pure_feature/${ProjName}/ /media/yy/10A4078410A40784/grad_proj/new_pure_feature/new_${ProjName}/

echo "generating vocabulary..."
python /media/yy/10A4078410A40784/grad_proj/code/makeVocab.py /media/yy/10A4078410A40784/grad_proj/new_pure_feature/new_${ProjName}/ /media/yy/10A4078410A40784/grad_proj/exp/dicts/${ProjName}.csv

find new_${ProjName} -name "*${Version_1}*" | xargs cp -t ${ProjName}_${Version_1}
find new_${ProjName} -name "*${Version_2}*" | xargs cp -t ${ProjName}_${Version_2}

rm -r ${ProjName}
rm -r new_${ProjName}

echo "generating samples..."
python /media/yy/10A4078410A40784/grad_proj/code/genSamples.py /media/yy/10A4078410A40784/grad_proj/new_pure_feature/${ProjName}_${Version_1}/ /media/yy/10A4078410A40784/grad_proj/exp/dicts/${ProjName}.csv /media/yy/10A4078410A40784/grad_proj/exp/dicts/${ProjName}_${Version_1}.
python /media/yy/10A4078410A40784/grad_proj/code/genSamples.py /media/yy/10A4078410A40784/grad_proj/new_pure_feature/${ProjName}_${Version_2}/ /media/yy/10A4078410A40784/grad_proj/exp/dicts/${ProjName}.csv /media/yy/10A4078410A40784/grad_proj/exp/dicts/${ProjName}_${Version_2}.

echo "generate samples done."
