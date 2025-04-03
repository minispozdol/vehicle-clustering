# vehicle-clustering

## Instructions for setup
In order to run this project, you will need an installation of Python, Jupyter and all associate libraries:
* Seaborn
* Numpy
* Pandas

A web browers should also be installed if the user vishes to see any of the data visualizations and graphs. 
Once Jupyter notebook in running and all libraries are successfully loaded onto the natebook, then there is an order of which file to run first before the others. For my workspace, the file "preprocessing.py" should run first before runnign the "final_evaluation.py" or the evaluationipynb. This will allow the raw csv file to be process into input for the algorithm.

## General purpose
This project is to take in unlabeled data and cluster them using three different algorithms K-Means, GMM, and K-Prototype. The output will then be evaluated against a table of verified data to see how the performance of the algorithms stack up with each other and the true data.
