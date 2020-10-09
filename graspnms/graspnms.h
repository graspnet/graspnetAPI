// Code Written by Minghao Gou

#ifndef GRASPNMS_H
#define GRASPNMS_H
#include<iostream>
#include<stdio.h>
#include<cstring>
#include<list>
#include<vector>
#include<algorithm>
#include<cmath>
#include"geo.cpp"
using namespace std;

struct arg_value
{
    unsigned int arg;
    double score;
};
class tuple_thresh
{
public:
    double translation_thresh,rotation_thresh;
    tuple_thresh();
    tuple_thresh(double,double);
    int smaller(tuple_thresh);
    void print_thresh();
};

class double_array
{
public:
    double* data;
    int r,c;
    double_array(int,int);
    double_array();
    ~double_array();
    void print_data();
};

double_array grasp_nms(double_array,tuple_thresh);

template <typename T> std::vector<unsigned int> reverse_argsort(const std::vector<T> &array);
tuple_thresh iou(double*,double*);

vector<unsigned int> argsort_grasp(const double_array&);

#endif