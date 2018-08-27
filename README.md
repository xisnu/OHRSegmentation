# Segmentation of Online handwriting data
A Neural Network based segmentation module for Bangla Online Handwriting data. This segmented substrokes can be clustered together. The data is kept in HDF format inside [Data](../Data/) directory. Every group of this HDF file is a word (Bangla script). Each word can have multiple strokes. Each stroke is one dataset under the group. Each stroke of N points is actually a Nx3 array. The first two dimensions contains the *X* and *Y* coordinates of the digital pen tip and the last dimension denotes a special segmentation mark. This last dimension can have two values,
- *1.0* - Means the pen is in touch of board
- *2.0* - Means a special segmentation point that is marked by mannual annotation

## Install
1. Tensorflow 1.5 and above
2. Numpy
3. H5py
4. PIL

## Execute
Two scripts can be executed here,
1. If ``predict_points_of_sample(nbfeat,"samplename",hdfin)`` of ``Segmentation_Point_Classifier.py`` is executed, then we get an offline image of that ``samplename`` in [Samples](../Samples/) directory. The image looks like ![segmented](https://github.com/xisnu/OHRSegmentation/blob/master/Samples/Akinchan34_AmdAnI.txt_predicted.png). The red points are natural end of stroke (writer lifted the pen tip off). The green points are segmentation points suggested by the Neural Net.
2. If ``generate_substroke_for_cluster(nbfeat,hdfin,hdfout)`` of ``Segmentation_Point_Classifier.py`` is executed then a new HDF file ``hdfout`` is created inside [Data](../Data/) directory. Each group of this new file is a segmented stroke
+ starts from start of stroke and ends in red point
+ starts from start of stroke and ends in green point
+ starts from a green point and ends in red point
+ starts from a green point and ends in another green point
