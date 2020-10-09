// Code Written by Minghao Gou

#ifndef GEO_H
#define GEO_H
#pragma once

#include <vector>
#include <cmath>
#include <iostream>
using namespace std;

class Mat3x3
{
public:
    friend ostream& operator<<(ostream&,Mat3x3);
    Mat3x3();
    Mat3x3(vector<double>);
    Mat3x3(vector<double>,vector<double>,vector<double>,int);
    // [[0 1 2]
    //  [3 4 5]
    //  [6 7 9]]
    double data[3][3];
    Mat3x3 matmul(Mat3x3);
    Mat3x3 T();
    double trace();
    double& e(int,int);
};

Mat3x3 rotation_array_to_matrix(double *);
Mat3x3 viewpoint_params_to_matrix(double,double,double,double);
vector<double> get_vector_3(double,double,double);
double norm(vector<double>);
vector<double> normal_form(vector<double>);
vector<double> cross(vector<double>,vector<double>);

#endif