#include <iostream>
#include <math.h>

using namespace std;

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
          euclideanDistances[(dataset_depth*zT)+zD] += abs(testArray[(zT * height * width) + (y * width) + x] - datasetArray[(zD * height * width) + (y * width) + x]);
        }
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

int main()
{
  int test_depth = 2;
  int data_depth = 5;
  int width = 3;
  int height = 3;
  int *test_test_tensor = new int[test_depth * width * height];
  int *test_data_tensor = new int[data_depth * width * height];

  float *test_output_array = new float[test_depth * data_depth];

  // Fill up test tensor.
  int counter = 0;
  for (int z = 0; z < test_depth; z++)
  {
    for (int x = 0; x < width; x++)
    {
      for (int y = 0; y < height; y++)
      {
        test_test_tensor[(z * width * height) + (y * width) + (x)] = ++counter;
      }
    }
  }
  // Fill up data tensor.
  counter = 0;
  for (int z = 0; z < data_depth; z++)
  {
    for (int x = 0; x < width; x++)
    {
      for (int y = 0; y < height; y++)
      {
        test_data_tensor[(z * width * height) + (y * width) + (x)] = ++counter;
      }
    }
  }

  // Calculate euclidean distances and place in the output tensor.
  serialEuclideanDistance(test_test_tensor, test_data_tensor, test_depth, data_depth, width, height, test_output_array);
  printTensor(test_test_tensor, test_depth, width, height);
  printTensor(test_data_tensor, data_depth, width, height);

  for(int i = 0; i<test_depth*data_depth; i++)
  {
    cout << test_output_array[i] <<" ";
  }
  

  return 0;

}