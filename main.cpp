#include "eigen-3.3.8/Eigen/Core"
#include "eigen-3.3.8/Eigen/Dense"
#include "fstream"
#include "iostream"
#include "iterator"
#include "math.h"
#include "string"
#include "vector"
using namespace std;
using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic, RowMajor> rMatrixXd;

const int layer1_input = 9;
const int layer1_output = 1600;
const int layer2_input = 1600;
const int layer2_output = 800;
const int layer3_input = 800;
const int layer3_output = 400;
const int layer4_input = 400;
const int layer4_output = 9;

double layer1_array[1600 * 9] = {0};
double layer1bias_array[1600] = {0};
double layer2_array[800 * 1600] = {0};
double layer2bias_array[800] = {0};
double layer3_array[400 * 800] = {0};
double layer3bias_array[400] = {0};
double layer4_array[9 * 400] = {0};
double layer4bias_array[9] = {0};

vector<double> model_vector;

void load_from_file(double *array, string path)
{
    ifstream file(path, ios::in);
    file.read((char *) array, sizeof(array) * 1600 * 9);
    file.close();
}

//void read_model(string path, int row, int col) {
rMatrixXd read_model(string path, int row, int col)
{
        if (row * col == layer1_input * layer1_output) {
            load_from_file(layer1_array, path);
            return Map<rMatrixXd>(layer1_array, layer1_output, layer1_input);
        }
        else if (row * col == layer1_output) {
            load_from_file(layer1bias_array, path);
            return Map<rMatrixXd>(layer1bias_array, layer1_output, 1);
        }
        else if (row * col == layer2_input * layer2_output) {
            load_from_file(layer2_array, path);
            return Map<rMatrixXd>(layer2_array, layer2_output, layer2_input);
        }
        else if (row * col == layer2_output) {
            load_from_file(layer1bias_array, path);
            return Map<rMatrixXd>(layer2bias_array, layer2_output, 1);
        }
        else if (row * col == layer3_input * layer3_output) {
            load_from_file(layer3_array, path);
            return Map<rMatrixXd>(layer3_array, layer3_output, layer3_input);
        }
        else if (row * col == layer3_output) {
            load_from_file(layer3bias_array, path);
            return Map<rMatrixXd>(layer3bias_array, layer3_output, 1);
        }
        else if (row * col == layer4_input * layer4_output) {
            load_from_file(layer4_array, path);
            return Map<rMatrixXd>(layer4_array, layer4_output, layer4_input);
        }
        else if (row * col == layer4_output) {
            load_from_file(layer4bias_array, path);
            return Map<rMatrixXd>(layer4bias_array, layer4_output, 1);
        }
}

double GeRelu(double x)
{
    return 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.44715 * pow(x, 3))));
}

int main()
{
    vector<int> weight_shape;
    vector<int> bias_shape;
    ifstream f("C:\\Users\\drsun\\CLionProjects\\DNN\\layer.txt", ios::in);
    int shape_number;
    int i = 0;
        while (f >> shape_number) {
            i++;
                if (i % 3 == 0 && i != 0) {
                    bias_shape.push_back(shape_number);
                }
                else {
                    weight_shape.push_back(shape_number);
                }
        }
    /*ostream_iterator<int> out_vector(std::cout, " ");
  cout << "bias" << endl;
  copy(bias_shape.begin(), bias_shape.end(), out_vector);
  cout << "weight" << endl;
  copy(weight_shape.begin(), bias_shape.end(), out_vector);*/
    f.close();

    /*double layer1_array[1600*9] = {0};
  ifstream model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.0.weight.bin",
  ios::in); model.read((char *) &layer1_array, sizeof layer1_array);
  //cout << model.gcount() << " bytes read\n";
  model.close();*/
    rMatrixXd layer1 = read_model("C:\\Users\\drsun\\CLionProjects\\DNN\\fc.0.weight.bin",
                                  layer1_output, layer1_input);

    //Matrix<double, 1600, 9> layer1;
    //layer1 = Map<Matrix<double, 1600, 9, RowMajor>>(layer1_array); // check pass
    //cout << layer1 << endl;
    Matrix<double, 9, 1> input;
    input << 1, 1, 1, 1, 1, 1, 1, 1, 1;

    cout << "used before GERELU " << (layer1 * input)(0, 0) << endl;//got problem
    cout << layer1.row(0) << endl;
    auto layer1_result = (layer1 * input).unaryExpr(&GeRelu);
    cout << "used after GERELU " << layer1_result(0, 0) << endl;
    cout << "result confirm" << 0.5 * 0.1022497 * (1 + tanh(sqrt(2 / M_PI) * (0.1022497 + 0.44715 * pow(0.1022497, 3))));
}