#include <iostream>
#include <nvjpeg.h>
#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <iterator>

#include <string.h> // strcmpi
#ifndef _WIN64
#include <sys/time.h> // timings
#include <unistd.h>
#endif
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include <cuda_runtime_api.h>
#include <nvjpeg.h>
#include <nppi_geometry_transforms.h>

//#include "imageResize.h"
//#include "nvjpegDecoder.h"

using namespace std;
//using namespace cv;

// CUDA will manage the locality of these variables, moving them to the device if necessary.
__managed__ int dev_test_depth;
__managed__ int dev_test_width;
__managed__ int dev_test_height;
__managed__ int dev_dataset_depth;
__managed__ int dev_dataset_width;
__managed__ int dev_dataset_height;

__global__ void devicePrintTensor(int *tensor, int width, int height, int depth)
{
  int counter3 = 0;
  for (int z = 0; z < depth; z++)
  {
    for (int x = 0; x < width; x++)
    {
      for (int y = 0; y < height; y++)
      {
        //cout << ++counter3 << " " << tensor[i][j][k] << "\n";
        printf("%d | %d\n", ++counter3, tensor[(height * width * z) + (width * y) * (x)]);
      }
    }
  }
}

void printTensor(int *input_tensor, int input_depth, int input_width, int input_height)
{
  for (int z = 0; z < input_depth; z++)
  {
    for (int x = 0; x < input_width; x++)
    {
      for (int y = 0; y < input_height; y++)
      {
        cout << input_tensor[(z * input_width * input_height) + (y * input_width) + (x)] << " ";
      }
      cout << "\n";
    }
    cout << "\n";
  }
}

__global__ void parallelEuclideanDistance(int *testArray, int *datasetArray, int test_depth, int dataset_depth, int width, int height, float *euclideanDistances)
{
  //int i = threadIdx.x + blockIdx.x * blockDim.x;
  //int j = threadIdx.y + blockIdx.y * blockDim.y;
  //int k = threadIdx.z + blockIdx.z * blockDim.z;
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  int zT = tId / (dataset_depth * width * height);
  int zD = (tId / (width * height)) % dataset_depth;
  int x = (tId / height) % width;
  int y = tId % height;

  if ((zD < dataset_depth) && (x < width) && (y < height))
    euclideanDistances[(dataset_depth * zT) + zD] += sqrt(pow(testArray[(zT * height * width) + (y * width) + x] - datasetArray[(zD * height * width) + (y * width) + x], 2));
}

void serialEuclideanDistance(int *testArray, int *datasetArray, int test_depth, int dataset_depth, int width, int height, float *euclideanDistances)
{
  for (int zT = 0; zT < test_depth; zT++)
  {
    for (int zD = 0; zD < dataset_depth; zD++)
    {
      for (int x = 0; x < width; x++)
      {
        for (int y = 0; y < height; y++)
        {
          //euclideanDistances[(zT * width) + x] += sqrt(pow(testArray[(zT * height * width) + (y * width) + x] - datasetArray[(zD * height * width) + (y * width) + x], 2));
          euclideanDistances[(dataset_depth * zT) + zD] += abs(testArray[(zT * height * width) + (y * width) + x] - datasetArray[(zD * height * width) + (y * width) + x]);
        }
      }
    }
  }
}

bool checkCudaDevices()
{
  int nDevices;
  cudaGetDeviceCount(&nDevices);

  if (nDevices > 0)
  {
    for (int i = 0; i < nDevices; i++)
    {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      printf("No of CUDA devices: %d\n", nDevices);
      printf("Device Number: %d\n", i);
      printf("  Device name: %s\n", prop.name);
      printf("  Memory Clock Rate (KHz): %d\n",
             prop.memoryClockRate);
      printf("  Memory Bus Width (bits): %d\n",
             prop.memoryBusWidth);
      printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
             2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }
  }
  return nDevices;
}

// *****************************************************************************
// reading input directory to file list
// -----------------------------------------------------------------------------
int readInput(const std::string &sInputPath, std::vector<std::string> &filelist)
{
  int error_code = 1;
  struct stat s;

  if (stat(sInputPath.c_str(), &s) == 0)
  {
    if (s.st_mode & S_IFREG)
    {
      filelist.push_back(sInputPath);
    }
    else if (s.st_mode & S_IFDIR)
    {
      // processing each file in directory
      DIR *dir_handle;
      struct dirent *dir;
      dir_handle = opendir(sInputPath.c_str());
      std::vector<std::string> filenames;
      if (dir_handle)
      {
        error_code = 0;
        while ((dir = readdir(dir_handle)) != NULL)
        {
          if (dir->d_type == DT_REG)
          {
            std::string sFileName = sInputPath + dir->d_name;
            filelist.push_back(sFileName);
          }
          else if (dir->d_type == DT_DIR)
          {
            std::string sname = dir->d_name;
            if (sname != "." && sname != "..")
            {
              readInput(sInputPath + sname + "/", filelist);
            }
          }
        }
        closedir(dir_handle);
      }
      else
      {
        std::cout << "Cannot open input directory: " << sInputPath << std::endl;
        return error_code;
      }
    }
    else
    {
      std::cout << "Cannot open input: " << sInputPath << std::endl;
      return error_code;
    }
  }
  else
  {
    std::cout << "Cannot find input path " << sInputPath << std::endl;
    return error_code;
  }

  return 0;
}

// *****************************************************************************
// check for inputDirExists
// -----------------------------------------------------------------------------
int inputDirExists(const char *pathname)
{
  struct stat info;
  if (stat(pathname, &info) != 0)
  {
    return 0; // Directory does not exists
  }
  else if (info.st_mode & S_IFDIR)
  {
    // is a directory
    return 1;
  }
  else
  {
    // is not a directory
    return 0;
  }
}

// *****************************************************************************
// check for getInputDir
// -----------------------------------------------------------------------------
int getInputDir(std::string &input_dir, const char *executable_path)
{
  int found = 0;
  if (executable_path != 0)
  {
    std::string executable_name = std::string(executable_path);
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    // Windows path delimiter
    size_t delimiter_pos = executable_name.find_last_of('\\');
    executable_name.erase(0, delimiter_pos + 1);

    if (executable_name.rfind(".exe") != std::string::npos)
    {
      // we strip .exe, only if the .exe is found
      executable_name.resize(executable_name.size() - 4);
    }
#else
    // Linux & OSX path delimiter
    size_t delimiter_pos = executable_name.find_last_of('/');
    executable_name.erase(0, delimiter_pos + 1);
#endif

    // Search in default paths for input images.
    std::string pathname = "";
    const char *searchPath[] = {
        "./images"};

    for (unsigned int i = 0; i < sizeof(searchPath) / sizeof(char *); ++i)
    {
      std::string pathname(searchPath[i]);
      size_t executable_name_pos = pathname.find("<executable_name>");

      // If there is executable_name variable in the searchPath
      // replace it with the value
      if (executable_name_pos != std::string::npos)
      {
        pathname.replace(executable_name_pos, strlen("<executable_name>"),
                         executable_name);
      }

      if (inputDirExists(pathname.c_str()))
      {
        input_dir = pathname + "/";
        found = 1;
        break;
      }
    }
  }
  return found;
}

// *****************************************************************************
// parse parameters
// -----------------------------------------------------------------------------
int findParamIndex(const char **argv, int argc, const char *parm)
{
  int count = 0;
  int index = -1;

  for (int i = 0; i < argc; i++)
  {
    if (strncmp(argv[i], parm, 100) == 0)
    {
      index = i;
      count++;
    }
  }

  if (count == 0 || count == 1)
  {
    return index;
  }
  else
  {
    std::cout << "Error, parameter " << parm
              << " has been specified more than once, exiting\n"
              << std::endl;
    return -1;
  }

  return -1;
}

int main(int argc, char **argv)
{
  // Check if we have CUDA GPU and also view some of its specs.
  //bool hasCudaDevice = checkCudaDevices();
  //int numSMs;
  //cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
  //printf("Your GPU has %d SMs.\n", numSMs);

  string datasetInputPath = "./dataset_resized/";
  string testInputPath = "./test/";
  string labelPath = "labels.txt";

  vector<string> datasetInputFiles;
  vector<string> testInputFiles;
  if (readInput(datasetInputPath, datasetInputFiles))
  {
    return 1;
  }

  if (readInput(testInputPath, testInputFiles))
  {
    return 1;
  }

  printf("%d test images found, %d dataset images found.\n", (int)testInputFiles.size(), (int)datasetInputFiles.size());
  // for (int i = 0; i < testInputFiles.size(); i++)
  // {
  //   cout << testInputFiles[i] << "\n";
  // }
  // for (int i = 0; i < datasetInputFiles.size(); i++)
  // {
  //   cout << datasetInputFiles[i] << "\n";
  // }

  vector<cv::Mat> testImages;
  vector<cv::Mat> datasetImages;

  // Read our labels and put them into string vector.
  vector<string> datasetLabels;
  ifstream in(labelPath.c_str());
  string line;
  while (getline(in, line))
  {
    datasetLabels.push_back(line);
  }

  // Read the test images into our OpenCV matrix as grayscale.
  for (int i = 0; i < testInputFiles.size(); i++)
  {
    testImages.push_back(cv::imread(testInputFiles[i], cv::IMREAD_GRAYSCALE));
  }
  //Read the dataset images into oue OpenCV matrix as grayscale.
  for (int i = 0; i < datasetInputFiles.size(); i++)
  {
    datasetImages.push_back(cv::imread(datasetInputFiles[i], cv::IMREAD_GRAYSCALE));
  }

  // Display test image.
  cv::imshow("Test Image", testImages[0]);

  // Apply Gaussian Blur to test images.
  vector<cv::Mat> GaussianTestImages;
  copy(testImages.begin(), testImages.end(), back_inserter(GaussianTestImages));
  for (int i = 0; i < testImages.size(); i++)
  {
    //cv::copyTo(testImages, GaussianTestImages, CV_8U);
    cv::GaussianBlur(GaussianTestImages[i], GaussianTestImages[i], cv::Size(5, 5), 1.5);
  }

  // Apply Gaussian Blur to dataset images.
  vector<cv::Mat> GaussianDatasetImages;
  copy(datasetImages.begin(), datasetImages.end(), back_inserter(GaussianDatasetImages));
  for (int i = 0; i < datasetImages.size(); i++)
  {
    //cv::copyTo(datasetImages, GaussianDatasetImages, CV_8U);
    cv::GaussianBlur(GaussianDatasetImages[i], GaussianDatasetImages[i], cv::Size(5, 5), 1.5);
  }

  // Apply Canny filter to test images.
  vector<cv::Mat> CannyTestImages;
  copy(GaussianTestImages.begin(), GaussianTestImages.end(), back_inserter(CannyTestImages));
  for (int i = 0; i < testImages.size(); i++)
  {
    //cv::copyTo(GaussianTestImages, CannyTestImages, CV_8U);
    cv::Canny(CannyTestImages[i], CannyTestImages[i], 100, 200);
  }

  // Apply Canny filter to dataset images.
  vector<cv::Mat> CannyDatasetImages = datasetImages;
  copy(GaussianDatasetImages.begin(), GaussianDatasetImages.end(), back_inserter(CannyDatasetImages));
  for (int i = 0; i < datasetImages.size(); i++)
  {
    //cv::copyTo(GaussianDatasetImages, CannyDatasetImages, CV_8U);
    cv::Canny(CannyDatasetImages[i], CannyDatasetImages[i], 100, 200);
  }

  // // Try Harris corner detection on test and dataset.
  // vector<cv::Mat> HarrisTestImages = testImages;
  // for (int i = 0; i < testImages.size(); i++)
  // {
  //   cv::cornerHarris(testImages[i], HarrisTestImages[i], 2, 3, 0.14);
  // }

  // vector<cv::Mat> HarrisDatasetImages = datasetImages;
  // for (int i = 0; i < datasetImages.size(); i++)
  // {
  //   cv::cornerHarris(datasetImages[i], HarrisDatasetImages[i], 2, 3, 0.14);
  // }

  // vector<cv::Mat> BilateralTestImages = testImages;
  // for (int i = 0; i < testImages.size(); i++)
  // {
  //   cv::bilateralFilter(testImages[i], BilateralTestImages[i], -1, 150, 150, cv::BORDER_ISOLATED);
  // }

  // vector<cv::Mat> BilateralDatasetImages = datasetImages;
  // for (int i = 0; i < datasetImages.size(); i++)
  // {
  //   cv::bilateralFilter(datasetImages[i], BilateralDatasetImages[i], -1, 150, 150, cv::BORDER_ISOLATED);
  // }

  // *****************************************************************************
  // Simple tests for sanity purposes
  // -----------------------------------------------------------------------------
  cv::imshow("Test Image + canny", CannyTestImages[0]);
  //cv::waitKey(0);

  // *****************************************************************************
  // Copy Data to GPU then process...
  // -----------------------------------------------------------------------------

  int test_depth = testImages.size();
  int dataset_depth = datasetImages.size();
  int test_width = testImages[0].rows;
  int dataset_width = datasetImages[0].rows;
  int test_height = testImages[0].cols;
  int dataset_height = datasetImages[0].cols;

  // Dynamically allocate heap memory to hold test image(s) pixel data.
  int *testTensor = new int[test_depth * test_width * test_height];

  // Dynamically allocate heap memory to hold dataset pixel data.
  int *datasetTensor = new int[dataset_depth * dataset_width * dataset_height];

  // Load pixels of test images into classic array.
  //The first level of the array corresponds to each image.
  for (int veci = 0; veci < testImages.size(); veci++)
  {
    for (int rowj = 0; rowj < testImages[0].rows; rowj++)
    {
      for (int colk = 0; colk < testImages[0].cols; colk++)
      {
        testTensor[(veci * test_width * test_height) + (rowj) + (colk * test_width)] = CannyTestImages[veci].at<int>(rowj, colk);
      }
    }
  }

  // Crude test to check if every single pixel is present in the test image array.
  /*   int counter = 0;
  for(int veci=0;veci<testImages.size();veci++)
  {
    for(int rowj=0;rowj<testImages[0].rows;rowj++)
    {
      for(int colk=0;colk<testImages[0].cols;colk++)
      {
        cout << ++counter << " | " << testTensor[veci][rowj][colk] << "\n";
      }
    }
  } */

  // Check a few pixels from the test image array.
  /*   for(int i=0; i < 4;i++)
  {
    for(int j=0;j<10;j++)
    {
      for(int k=0;k<10;k++)
      {
        cout << testTensor[i][j][k];
      }
    }
  } */

  // Load pixels of dataset images into classic array.
  // We should also do canny version
  //The first level of the array corresponds to each image.
  for (int vecl = 0; vecl < datasetImages.size(); vecl++)
  {
    for (int rowm = 0; rowm < datasetImages[0].rows; rowm++)
    {
      for (int coln = 0; coln < datasetImages[0].cols; coln++)
      {
        datasetTensor[(vecl * dataset_width * dataset_height) + (rowm) + (coln * dataset_width)] = CannyDatasetImages[vecl].at<int>(rowm, coln);
        // To traverse 1D array in 3D fashion, index = height*width*z+width*y+x.
        //flatDatasetTensor[(datasetImages[0].cols*datasetImages[0].rows*vecl)+(datasetImages[0].rows*coln)+(rowm)] = datasetImages[vecl].at<int>(rowm, coln);
      }
    }
  }

  // Crude test to check if every single pixel is present in the dataset image array.
  /*   int counter = 0;
  for(int veci=0;veci<datasetImages.size();veci++)
  {
    for(int rowj=0;rowj<datasetImages[0].rows;rowj++)
    {
      for(int colk=0;colk<datasetImages[0].cols;colk++)
      {
        cout << ++counter << " | " << datasetTensor[veci][rowj][colk] << "\n";
      }
    }
  } */

  // Check a few pixels from the dataset image array.
  /*   for(int l = 0; l < 10; l++)
  {
    for(int m=0;m<10;m++)
    {
      for(int n=0;n<10;n++)
      {
        cout << datasetTensor[l][m][n];
      }
    }
  } */

  // *****************************************************************************
  // Test accuracy of eulidean function...
  // -----------------------------------------------------------------------------
  //int counterz =0;
  // for(int i = 0; i > testImages.size()*testImages[0].rows*testImages[0].cols; i++)
  // {
  //   cout << testTensor[i] << "\n";
  // }

  //cout << "Now for da main sauce..." << "\n";

  // *****************************************************************************
  // Time to work the gpu...
  // -----------------------------------------------------------------------------
  // int *dev_testTensor;
  // int *dev_datasetTensor;
  // float *dev_finalTensor;

  // dev_test_depth = test_depth;
  // dev_test_width = test_width;
  // dev_test_height = test_height;
  // dev_dataset_depth = dataset_depth;
  // dev_dataset_width = dataset_width;
  // dev_dataset_height = dataset_height;

  // // Allocate variables similar to the serial processing on the device
  // cudaMalloc((void **)&dev_testTensor, test_depth * test_width * test_height * sizeof(int));
  // cudaMalloc((void **)&dev_datasetTensor, dataset_depth * dataset_width * dataset_height * sizeof(int));
  // cudaMalloc((void **)&dev_finalTensor, test_depth * dataset_depth * sizeof(float));

  // // Copy data from host to device.
  // cudaMemcpy(dev_testTensor, testTensor, test_depth * test_width * test_height * sizeof(int), cudaMemcpyHostToDevice);
  // cudaMemcpy(dev_datasetTensor, datasetTensor, dataset_depth * dataset_width * dataset_height * sizeof(int), cudaMemcpyHostToDevice);

  // //devicePrintTensor<<<1, 1>>>(dev_testTensor, dev_test_width, dev_test_height, dev_test_depth);
  // parallelEuclideanDistance<<<4, (numSMs * 64 * 2)>>>(dev_testTensor, dev_datasetTensor, dev_test_depth, dev_dataset_depth, dev_test_width, dev_test_height, dev_finalTensor);
  // cudaDeviceSynchronize();

   float *host_finalTensor = new float[test_depth * dataset_depth]; // We will copy results of euclidean distance from the GPU to this array.
  // // Copy results back from device to host.
  // cudaMemcpy(host_finalTensor, dev_finalTensor, test_depth * dataset_depth * sizeof(float), cudaMemcpyDeviceToHost);

  serialEuclideanDistance(testTensor, datasetTensor, test_depth, dataset_depth, test_width, test_height, host_finalTensor);
  // for(int i = 0; i<test_depth*dataset_depth; i++)
  // {
  //   cout << host_finalTensor[i] << " ";
  // }
  cout << endl;

  vector<float> resultVect;
  resultVect.assign(host_finalTensor, host_finalTensor + (test_depth * dataset_depth));
  cout << "Result Vector Length = " << resultVect.size() << endl;
  float mnDistance = *min_element(resultVect.begin(), resultVect.end());
  cout << "Minimum Distance = " << mnDistance << "\n";

  std::vector<float>::iterator indexIterator;
  indexIterator = find(resultVect.begin(), resultVect.end(), mnDistance);
  cout << "Index of Minimum Distance: " << indexIterator - resultVect.begin() << "\n";
  cout << "Filename of Minimum Distance: " << datasetInputFiles[indexIterator - resultVect.begin()] << "\n";
  cout << "Prediction is: " << datasetLabels[indexIterator - resultVect.begin()] << "\n";
  cout << "Press any key to quit.\n";

  // for(int i =0; i<dataset_depth; i++)
  // {
  //   cout << datasetLabels[i] <<"\n";
  // }

  cv::Mat closestMatch = cv::imread(datasetInputFiles[indexIterator - resultVect.begin()], cv::IMREAD_GRAYSCALE);
  cv::imshow("Closest Match", closestMatch);
  cv::imshow("Closest Match + canny", datasetImages[indexIterator - resultVect.begin()]);
  //cv::imshow("Closest Match + canny", datasetImages[indexIterator - resultVect.begin()]);
  cv::waitKey(0);

  // *****************************************************************************
  // Test to check parallel euclidean distance function.
  // -----------------------------------------------------------------------------
  // //test test tensor
  // int *exp_tensora = new int[2 * 3 * 3];
  // int counta = 0;
  // for (int z = 0; z < 2; z++)
  //   for (int x = 0; x < 3; x++)
  //     for (int y = 0; y < 3; y++)
  //       exp_tensora[(z * 3 * 3) + (y * 3) + (x)] = ++counta;

  // //test dataset tensor
  // int *exp_tensorb = new int[5 * 3 * 3];
  // counta = 0;
  // for (int z = 0; z < 5; z++)
  //   for (int x = 0; x < 3; x++)
  //     for (int y = 0; y < 3; y++)
  //       exp_tensorb[(z * 3 * 3) + (y * 3) + (x)] = ++counta;

  // float *e_distances = new float[2 * 5];
  // float *test_host_finalTensor = new float[2 * 5];
  // printTensor(exp_tensora, 2, 3, 3);
  // printTensor(exp_tensorb, 5, 3, 3);
  // serialEuclideanDistance(exp_tensora, exp_tensorb, 2, 5, 3, 3, e_distances);
  // for (int i = 0; i < 2 * 5; i++)
  // {
  //   cout << "distances: " << e_distances[i] << "\n";
  // }

  // // Allocate variables similar to the serial processing on the device
  // cudaMalloc((void **)&dev_testTensor, 2 * 3 * 3 * sizeof(int));
  // cudaMalloc((void **)&dev_datasetTensor, 5 * 3 * 3 * sizeof(int));
  // cudaMalloc((void **)&dev_finalTensor, 2 * 5 * sizeof(float));

  // // Copy data from host to device.
  // cudaMemcpy(dev_testTensor, exp_tensora, 2 * 3 * 3 * sizeof(int), cudaMemcpyHostToDevice);
  // cudaMemcpy(dev_datasetTensor, exp_tensorb, 5 * 3 * 3 * sizeof(int), cudaMemcpyHostToDevice);

  // parallelEuclideanDistance<<<4, (numSMs * 64)>>>(dev_testTensor, dev_datasetTensor, 2, 5, 3, 3, dev_finalTensor);
  // cudaDeviceSynchronize();

  // // Copy result back to host.

  // cudaMemcpy(test_host_finalTensor, dev_finalTensor, 2 * 5 * sizeof(float), cudaMemcpyDeviceToHost);

  // for (int i = 0; i < 2 * 5; i++)
  // {
  //   cout << "distances from gpu: " << test_host_finalTensor[i] << "\n";
  // }

  return 0;
}
