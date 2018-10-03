echo "input proj & version:"
read proj

mkdir $proj

python /media/yy/10A4078410A40784/grad_proj/code/utils/selectFile.py \
/media/yy/10A4078410A40784/grad_proj/exp/defectInfo/${proj}.csv \
/media/yy/10A4078410A40784/grad_proj/exp/cross_feature/${proj}/ \
/media/yy/10A4078410A40784/grad_proj/exp/cross_select_feature/${proj}/

python /media/yy/10A4078410A40784/grad_proj/code/fileLabel.py \
/media/yy/10A4078410A40784/grad_proj/exp/defectInfo/${proj}.csv \
/media/yy/10A4078410A40784/grad_proj/exp/cross_select_feature/${proj}/
