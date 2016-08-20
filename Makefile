OPENCV := /usr/local/Cellar/opencv/2.4.13

all:
	clang-omp++ hyp.cpp -g -std=c++11 -I$(OPENCV)/include -L$(OPENCV)/lib -lopencv_core -lopencv_highgui -lopencv_nonfree -lopencv_features2d -lopencv_calib3d -lopencv_flann -lglog -fopenmp -o hyp
