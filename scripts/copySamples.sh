#!/bin/bash

echo "input project name:"
read ProjName
echo "input version:"
read Version


echo "is k fixed?"
read isFixed

if [ $isFixed = "no" ]
then
	for i in 2 3 4 5 6 7 8 9 10
	do
		echo "this is $i/10 times..."
		rm -r /media/yy/10A4078410A40784/grad_proj/new_pure_feature/${ProjName}_${Version}
		cd /media/yy/10A4078410A40784/grad_proj/pure_feature/
		cp ${ProjName}_${Version} -rt ../new_pure_feature/
		echo "copying samples..."
		python /media/yy/10A4078410A40784/grad_proj/code/copySamples.py /media/yy/10A4078410A40784/grad_proj/new_pure_feature/${ProjName}_${Version}/ $i
		echo "calculating the results..."
		cd /media/yy/10A4078410A40784/grad_proj/new_pure_feature/${ProjName}_${Version}
		total=`ls | wc -l`
		buggy=`ls *bug_* | wc -l`
		echo "$total files, $buggy is buggy."
		echo "scale=2; $buggy / $total" | bc
	done
else
	echo "choose your k:"
	read k
	rm -r /media/yy/10A4078410A40784/grad_proj/new_pure_feature/${ProjName}_${Version}
	cd /media/yy/10A4078410A40784/grad_proj/pure_feature/
	cp ${ProjName}_${Version} -rt ../new_pure_feature/
	python /media/yy/10A4078410A40784/grad_proj/code/copySamples.py /media/yy/10A4078410A40784/grad_proj/new_pure_feature/${ProjName}_${Version}/ $k
	echo "calculating the results..."
	cd /media/yy/10A4078410A40784/grad_proj/new_pure_feature/${ProjName}_${Version}
	total=`ls | wc -l`
	buggy=`ls *bug_* | wc -l`
	echo "$total files, $buggy is buggy."
	echo "scale=2; $buggy / $total" | bc
fi