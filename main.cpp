#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
using namespace std;
struct net
{ // 神經元
    double w[2]; // 權重
    double bais; // 閥值
    double sigmoid_value; // 神經元輸出
    double error_value;   // 神經元回傳錯誤值
};
double **create_newspace(int r, int c)
{ // 建立動態指標
    double **newspace = NULL;
    newspace = new double *[r];
    for (int i = 0; i < r; i++)
        newspace[i] = new double[c];
    return newspace;
}
net *create_net(int net_num, int weight_num, double **init_w)
{ // 建立神經元的原始參數
    net *net_ptr = new net[net_num];
    for (int i = 0; i < net_num; i++)
    {
        for (int j = 0; j < weight_num - 1; j++)
        {
            net_ptr[i].bais = init_w[i][weight_num - 1];
            net_ptr[i].w[j] = init_w[i][j];
        }
    }
    return net_ptr;
}
double Sigmoid(double in_value)
{ // Sigmoid 函數
    double out_value = 1 / (1 + exp(-in_value));
    return out_value;
}
double Sigmoid_der(double in_value)
{ // Sigmoid 導函數
    double out_value_der = in_value * (1 - in_value);
    return out_value_der;
}
double loss_function(double target, double pre_value)
{ // loss_function：最小平方法
    double loss = (1.0 / 2.0) * pow(target - pre_value, 2.0);
    return loss;
}
double loss_function_der(double target, double pre_value)
{ // 最小平方法取導數
    double loss_der = target - pre_value;
    return loss_der;
}
net *net_forward(double *data, net *NET)
{ // 前向傳播
    int i, j;
    for (i = 0; i < 2; i++)
    { // 隱藏層神經元
        for (j = 0; j < 2; j++)
            NET[i].sigmoid_value += data[j] * NET[i].w[j]; // 接受輸入
        NET[i].sigmoid_value = Sigmoid(NET[i].sigmoid_value - NET[i].bais); // 輸出預測
    }
    for (i = 0; i < 2; i++) // 輸出層神經元
        NET[2].sigmoid_value += NET[i].sigmoid_value * NET[2].w[i]; // 接受輸入
    NET[2].sigmoid_value = Sigmoid(NET[2].sigmoid_value - NET[2].bais); // 輸出預測
    return NET;
}
net *net_back(double *Data, net *NET)
{ // 倒傳遞
    NET[2].error_value = loss_function_der(Data[2], NET[2].sigmoid_value); // 從輸出層往回推
    NET[2].error_value *= Sigmoid_der(NET[2].sigmoid_value); // Sigmoid 導函數
    for (int i = 1; i >= 0; i--)
    { // 隱藏層往回推
        NET[i].error_value = Sigmoid_der(NET[i].sigmoid_value); // Sigmoid 導函數
        NET[i].error_value *= NET[2].w[i] * NET[2].error_value; // 每個權重的錯誤值
    }
    return NET;
}
double **cal_delta_weight(double *Data, net *NET, double learning_rate, double **delta_w)
{ // 計算權重修正量
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            if (i != 2) // 隱藏層修正權重與閥值
                delta_w[i][j] += learning_rate * NET[i].error_value * Data[j];
            else // 輸出層修正權重與閥值
                delta_w[i][j] += learning_rate * NET[i].error_value * NET[j].sigmoid_value;
        }
        delta_w[i][2] += -(learning_rate)*NET[i].error_value;
    }
    return delta_w;
}
double **renew_weight_bais(double **old_weight, double **delat_weight)
{ //更新權重與閥值
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
            old_weight[i][j] += delat_weight[i][j];
    }
    return old_weight;
}
int main()
{
    //初始設置
    int i, j, iteration = 0, data_row_num = 4, net_num = 3, weight_num = 3;
    double LearnR = 10.0, Error = 0.05, T_Y_error = 1.0; // 停止條件
    double **Data = NULL, **initW = NULL, **deltaW = NULL, **newW = NULL;
    double data[data_row_num][weight_num]={{-1,-1,1},{-1,1,0},{1,-1,0},{1,1,1}}; // AND
    //double data[data_row_num][weight_num] = {{-1, -1, 0}, {-1, 1, 1}, {1, -1, 1}, {1, 1, 1}}; // OR  double LearnR = 2.0, Error = 0.05, T_Y_error = 1.0; // 停止條件
    //double data[data_row_num][weight_num] = {{-1, -1, 0}, {-1, 1, 1}, {1, -1, 1}, {1, 1, 0}}; // XOR double LearnR = 10.0, Error = 0.005, T_Y_error = 1.0; // 停止條件
    double init_w[net_num][weight_num] = {{1.0, -1.0, 1.0}, {-1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}};
    // 建立data矩陣
    Data = create_newspace(data_row_num, weight_num); 
    for (i = 0; i < data_row_num; i++)
    {
        for (j = 0; j < weight_num; j++)
            Data[i][j] = data[i][j];
    }
    // 初始權重設置
    initW = create_newspace(net_num, weight_num);
    deltaW = create_newspace(net_num, weight_num);
    for (i = 0; i < net_num; i++)
    {
        for (j = 0; j < weight_num; j++)
        {
            initW[i][j] = init_w[i][j];
            deltaW[i][j] = 0.0;
        }
    }
    while (T_Y_error > Error)
    { // 停止條件：整體錯誤值小於0.05
        T_Y_error = 0.0;
        //每項輸入訓練神經元
        for (i = 0; i < data_row_num; i++)
        {
            net *NET = NULL, *OUTPUT = NULL, *BACK = NULL;
            NET = create_net(net_num, weight_num, initW); // 建立隱藏層，批量梯度下降維持同一權重
            OUTPUT = net_forward(Data[i], NET); // 前向傳播
            cout << "output=" << OUTPUT[2].sigmoid_value << endl;
            T_Y_error += loss_function(Data[i][2], OUTPUT[2].sigmoid_value); // 計算損失
            cout << "NO." << iteration << "，loss=" << T_Y_error << endl;
            BACK = net_back(Data[i], OUTPUT); // 倒傳遞
            deltaW = cal_delta_weight(Data[i], BACK, LearnR, deltaW); // 計算權重與閥值修正量
        }
        printf("---------------------------OK----------------------------------\n");
        initW = renew_weight_bais(initW, deltaW); //更新權重與閥值
        iteration++;
    }
    return 0;
}