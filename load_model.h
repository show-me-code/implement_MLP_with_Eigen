//
// Created by drsun on 2023/2/13.
//
#include "eigen-3.3.8/Eigen/Core"
#include "eigen-3.3.8/Eigen/Dense"
#include "fstream"
#include "iostream"
#include "iterator"
#include "math.h"
#include "omp.h"
#include "string"
#include "vector"

using namespace std;
using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic, RowMajor> rMatrixXd;//define layer matrix

//for further use and prevent confuse,  define 3 struct model
struct model {
    rMatrixXd layer0;
    rMatrixXd bias0;
    rMatrixXd layer2;
    rMatrixXd bias2;
    rMatrixXd layer4;
    rMatrixXd bias4;
    rMatrixXd layer6;
    rMatrixXd bias6;
    rMatrixXd layer8;
    rMatrixXd bias8;
};


const int layer0_input = 23;
const int layer0_output = 3200;
const int layer2_input = 3200;
const int layer2_output = 1600;
const int layer4_input = 1600;
const int layer4_output = 800;
const int layer6_input = 800;
const int layer6_output = 400;
const int layer8_input = 400;
const int layer8_output = 23;

double layer0_array[layer0_output * layer0_input] = {0};
double layer0bias_array[layer0_output] = {0};
double layer2_array[layer2_output * layer2_input] = {0};
double layer2bias_array[layer2_output] = {0};
double layer4_array[layer4_output * layer4_input] = {0};
double layer4bias_array[layer4_output] = {0};
double layer6_array[layer6_output * layer6_input] = {0};
double layer6bias_array[layer6_output] = {0};
double layer8_array[layer8_output * layer8_input] = {0};
double layer8bias_array[layer8_output] = {0};


void load_from_file(double *array, string path, int read_size)
{
    ifstream file(path, ios::in);
    file.read((char *) array, sizeof(array) * read_size);
    file.close();
}

rMatrixXd read_model(string path, int row, int col, int batch_size)
{
    if (row * col == layer0_input * layer0_output)
        {
            load_from_file(layer0_array, path, row * col);
            return Map<rMatrixXd>(layer0_array, layer0_output, layer0_input);
        }
    else if (row * col == layer0_output * batch_size)
        {
            load_from_file(layer0bias_array, path, row);
            rMatrixXd bias = Map<rMatrixXd>(layer0bias_array, layer0_output, 1);
            rMatrixXd result_bias(layer0_output, batch_size);
#pragma omp parallel for
                for (int i = 0; i < batch_size; ++i)
                    {
                        result_bias.col(i) = bias.col(0);
                    }
            return result_bias;
        }
    else if (row * col == layer2_input * layer2_output)
        {
            load_from_file(layer2_array, path, row * col);
            return Map<rMatrixXd>(layer2_array, layer2_output, layer2_input);
        }
    else if (row * col == layer2_output * batch_size)
        {
            load_from_file(layer2bias_array, path, row);
            rMatrixXd bias = Map<rMatrixXd>(layer2bias_array, layer2_output, 1);
            rMatrixXd result_bias(layer2_output, batch_size);
#pragma omp parallel for

                for (int i = 0; i < batch_size; ++i)
                    {
                        result_bias.col(i) = bias.col(0);
                    }

            return result_bias;
        }
    else if (row * col == layer4_input * layer4_output)
        {
            load_from_file(layer4_array, path, row * col);
            return Map<rMatrixXd>(layer4_array, layer4_output, layer4_input);
        }
    else if (row * col == layer4_output * batch_size)
        {
            load_from_file(layer4bias_array, path, row);
            rMatrixXd bias = Map<rMatrixXd>(layer4bias_array, layer4_output, 1);
            rMatrixXd result_bias(layer4_output, batch_size);
#pragma omp parallel for

                for (int i = 0; i < batch_size; ++i)
                    {
                        result_bias.col(i) = bias.col(0);
                    }

            return result_bias;
        }
    else if (row * col == layer6_input * layer6_output)
        {
            load_from_file(layer6_array, path, row * col);
            return Map<rMatrixXd>(layer6_array, layer6_output, layer6_input);
        }
    else if (row * col == layer6_output * batch_size)
        {
            load_from_file(layer6bias_array, path, row);
            rMatrixXd bias = Map<rMatrixXd>(layer6bias_array, layer6_output, 1);
            rMatrixXd result_bias(layer6_output, batch_size);
#pragma omp parallel for
            for (int i = 0; i < batch_size; ++i)
                {
                    result_bias.col(i) = bias.col(0);
                }

            return result_bias;
        }
    else if (row * col == layer8_input * layer8_output)
        {
            load_from_file(layer8_array, path, row * col);
            return Map<rMatrixXd>(layer8_array, layer8_output, layer8_input);
        }
    else if (row * col == layer8_output * batch_size)
        {
            load_from_file(layer8bias_array, path, row);
            rMatrixXd bias = Map<rMatrixXd>(layer8bias_array, layer8_output, 1);
            rMatrixXd result_bias(layer8_output, batch_size);
#pragma omp parallel for
            for (int i = 0; i < batch_size; ++i)
                {
                    result_bias.col(i) = bias.col(0);
                }

            return result_bias;
        }
}


void load_model0(model *m0, int batch_size)
{
    //when input size != batch_size
    m0->layer0 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.0.weight_model0.bin", layer0_output, layer0_input, 0);
    m0->bias0 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.0.bias_model0.bin", layer0_output, batch_size, batch_size);
    m0->layer2 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.2.weight_model0.bin", layer2_output, layer2_input, 0);
    m0->bias2 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.2.bias_model0.bin", layer2_output, batch_size, batch_size);
    m0->layer4 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.4.weight_model0.bin", layer4_output, layer4_input, 0);
    m0->bias4 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.4.bias_model0.bin", layer4_output, batch_size, batch_size);
    m0->layer6 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.6.weight_model0.bin", layer6_output, layer6_input, 0);
    m0->bias6 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.6.bias_model0.bin", layer6_output, batch_size, batch_size);
    m0->layer8 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.8.weight_model0.bin", layer8_output, layer8_input, 0);
    m0->bias8 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.8.bias_model0.bin", layer8_output, batch_size, batch_size);
}

void load_model1(model *m1, int batch_size)
{
    m1->layer0 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.0.weight_model1.bin", layer0_output, layer0_input, 0);
    m1->bias0 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.0.bias_model1.bin", layer0_output, batch_size, batch_size);
    m1->layer2 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.2.weight_model1.bin", layer2_output, layer2_input, 0);
    m1->bias2 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.2.bias_model1.bin", layer2_output, batch_size, batch_size);
    m1->layer4 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.4.weight_model1.bin", layer4_output, layer4_input, 0);
    m1->bias4 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.4.bias_model1.bin", layer4_output, batch_size, batch_size);
    m1->layer6 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.6.weight_model1.bin", layer6_output, layer6_input, 0);
    m1->bias6 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.6.bias_model1.bin", layer6_output, batch_size, batch_size);
    m1->layer8 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.8.weight_model1.bin", layer8_output, layer8_input, 0);
    m1->bias8 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.8.bias_model0.bin", layer8_output, batch_size, batch_size);
}

void load_model2(model *m2, int batch_size)
{
    m2->layer0 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.0.weight_model2.bin", layer0_output, layer0_input, 0);
    m2->bias0 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.0.bias_model2.bin", layer0_output, batch_size, batch_size);
    m2->layer2 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.2.weight_model2.bin", layer2_output, layer2_input, 0);
    m2->bias2 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.2.bias_model2.bin", layer2_output, batch_size, batch_size);
    m2->layer4 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.4.weight_model2.bin", layer4_output, layer4_input, 0);
    m2->bias4 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.4.bias_model2.bin", layer4_output, batch_size, batch_size);
    m2->layer6 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.6.weight_model2.bin", layer6_output, layer6_input, 0);
    m2->bias6 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.6.bias_model2.bin", layer6_output, batch_size, batch_size);
    m2->layer8 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.8.weight_model2.bin", layer8_output, layer8_input, 0);
    m2->bias8 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.8.bias_model0.bin", layer8_output, batch_size, batch_size);
}