# defect prediction

1.  download source and defect info.
2.  extract features from source code using Eclipse JDT.
3.  select feature files that contained in defectInfo.csv.
4.  label feature files according to defectInfo.csv.
5.  remove type name for data.
6.  remove noise for both train and test data.
7.  resample train data.
8.  remove tokens appear less than three times.
9.  make vocabularies.
10.  generate sample and label files for train and test data.
11. train dbn with train data.
12. gengerate semantic features for train data and test data.
13. build classifier and test.
