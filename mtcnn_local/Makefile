all: clean mtcnn_local

LIBS = -lopencv_dnn -lopencv_imgcodecs -lopencv_imgproc  -lopencv_core
CXXFLAGS = $(LIBS) -L/usr/local/lib64

CC=mtcnn_detector.cc opencv_detector.cc mtcnn_local.cc

mtcnn_local: $(CC)
	g++ $(CXXFLAGS) $^ -o  $@ -std=c++11

clean:
	rm mtcnn_local -f
