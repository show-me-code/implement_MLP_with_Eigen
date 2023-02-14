//
// Created by drsun on 2023/2/13.
//
#include "chrono"
#include "load_model.h"

double GeRelu(double x)
{
    return 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.44715 * pow(x, 3))));
}

rMatrixXd inference(model m, rMatrixXd input)
{
    auto layer0_result = (m.layer0 * input + m.bias0).unaryExpr(&GeRelu);
    auto layer2_result = (m.layer2 * layer0_result + m.bias2).unaryExpr(&GeRelu);
    auto layer4_result = (m.layer4 * layer2_result + m.bias4).unaryExpr(&GeRelu);
    auto layer6_result = (m.layer6 * layer4_result + m.bias6).unaryExpr(&GeRelu);
    auto layer8_result = (m.layer8 * layer6_result + m.bias8).unaryExpr(&GeRelu);
    return layer8_result;
}

int main()
{
    model m0;
    model m1;
    model m2;
    rMatrixXd input;
    input.setRandom(23, 10000);

    load_model0(&m0, input.cols());
    load_model1(&m1, input.cols());
    load_model2(&m2, input.cols());

    //cout << m0.bias8 << endl;
    cout << "start" << endl;
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    rMatrixXd result = inference(m0, input);
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    //std::chrono::duration<double> processingTime = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " ms consumed" << endl;
    //cout << result << endl;
}