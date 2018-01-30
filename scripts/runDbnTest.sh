#!/bin/bash

# remove the logs generated last time
rm -r /home/yy/.yadlt/models
rm -r /home/yy/.yadlt/logs

echo "input project name:"
read ProjName
echo "input version 1:"
read Version_1
echo "input version 2:"
read Version_2

for i in 1 2 3 4 5
do
	echo "this is $i/5 times..."
	python /media/yy/10A4078410A40784/grad_proj/code/run_dbn.py /media/yy/10A4078410A40784/grad_proj/exp/dicts/${ProjName}_${Version_1}.Sample.npy /media/yy/10A4078410A40784/grad_proj/exp/dicts/${ProjName}_${Version_1}.Label.npy /media/yy/10A4078410A40784/grad_proj/exp/dicts/${ProjName}_${Version_2}.Sample.npy /media/yy/10A4078410A40784/grad_proj/exp/dicts/${ProjName}_${Version_2}.Label.npy /media/yy/10A4078410A40784/grad_proj/exp/dicts/${ProjName}_${Version_1}.DBN.npy /media/yy/10A4078410A40784/grad_proj/exp/dicts/${ProjName}_${Version_2}.DBN.npy
	python /media/yy/10A4078410A40784/grad_proj/code/logistic.py /media/yy/10A4078410A40784/grad_proj/exp/dicts/${ProjName}_${Version_1}.DBN.npy /media/yy/10A4078410A40784/grad_proj/exp/dicts/${ProjName}_${Version_1}.Label.npy /media/yy/10A4078410A40784/grad_proj/exp/dicts/${ProjName}_${Version_2}.DBN.npy /media/yy/10A4078410A40784/grad_proj/exp/dicts/${ProjName}_${Version_2}.Label.npy
done

echo "*********refer to tensorboard**********"
supertux2
