# geto-CUDA-kNN

Basic implementation of k-Nearest Neighbour Algorithm with CUDA acceleration for euclidean distance calculation. Borrows code from nvJPEG in NVIDIA/CUDALibrarySamples that can be found at https://github.com/NVIDIA/CUDALibrarySamples

#######Linux#########
*Tested on Ubuntu 20.04.3 LTS.
*You need only compile the kNN.cu source file with nvcc compiler.
serialknn.cu is there for convenience to run comparisons with serial execution.
*when knn.out binary is run, it checks dataset_resised directory for dataset images by default and test directory for test image.All images must have the same resolution or bad things will happen.
*imageResize is a means to conveniently resise jpeg images, the source can be found in CUDALibrarySamples directory.
*default label file is 'labels.txt

*To compile...
1. Make sure cuda toolkit and runtime are installed.(nvidia-cuda-dev, nvidia-cuda-toolkit)
2. Also ensure opencv libraries are present on your system(libopencv-dev).
3. Run the following line in the root of the project:
nvcc --include-path /usr/include/opencv4/ kNN.cu -o knn.out -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui



######Windows######
*Tested on Windows 10 and Visual Studio 2019.
*CUDA toolkit 11.4.2
*OpenCV 4.5.5

*Ensure OpenCV is installed and included in PATH as well as include path of visual studio
*Create CUDA project in Visual Studio and build project using kNN.cu source file.

