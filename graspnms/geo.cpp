// Code Written by Minghao Gou

#include "geo.h"
#include <cmath>
#include <cstring>
using namespace std;

ostream& operator<<(ostream& out,Mat3x3 m)
{
    out<<"Matrix 3x3:\n";
    for (int i=0;i<3;i++)
    {
        for(int j=0;j<3;j++)
        {
            out <<m.data[i][j]<<" ";
        }
        out<<endl;
    }
    return out;
}

Mat3x3::Mat3x3()
{
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            (this->data)[i][j] = 0;
        }
    }
}

Mat3x3::Mat3x3(vector<double> v)
{
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            (this->data)[i][j] = v[i * 3 + j];
        }
    }
}

Mat3x3::Mat3x3(vector<double> v0,vector<double> v1,vector<double> v2,int type)
{
    for(int i = 0;i < 3; i++)
    {
        if(type == 1) // column vector
        {
            (this->data)[i][0] = v0[i];
            (this->data)[i][1] = v1[i];
            (this->data)[i][2] = v2[i];
        }
        else if(type == 0) // row vector
        {
            (this->data)[0][i] = v0[i];
            (this->data)[1][i] = v1[i];
            (this->data)[2][i] = v2[i];
        }
    }
}

double& Mat3x3::e(int i,int j)
{
    return this->data[i][j];
}

double Mat3x3::trace()
{
    return this -> data[0][0] + this -> data[1][1] + this -> data[2][2];
}

Mat3x3 Mat3x3::T()
{
    Mat3x3 m;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            (m.data)[i][j] = this -> data[j][i];
        }
    }
    return m;
}

Mat3x3 Mat3x3::matmul(Mat3x3 m)
{
    // using ikj will be faster
    // in fact, for 3x3 matrix it doesn't matter
    Mat3x3 rm;
    for(int i = 0;i < 3;i++)
    {
        for(int k = 0;k < 3;k++)
        {
            int j;
            for(j = 0; j < 3; j++)
            {
                rm.data[i][j] += this->data[i][k]*m.data[k][j];
            }
        }
    }
    return rm;
}

vector<double> get_vector_3(double d1,double d2,double d3)
{
    vector<double> v;
    v.reserve(3);
    v.push_back(d1);
    v.push_back(d2);
    v.push_back(d3);
    return v;
}

Mat3x3 rotation_array_to_matrix(double *rotation_start)
{
    Mat3x3 rotation_matrix;
    memcpy(&(rotation_matrix.data[0][0]), rotation_start, sizeof(double) * 9);
    return rotation_matrix;
}

Mat3x3 viewpoint_params_to_matrix(double appx,double appy,double appz,double angle)
{
    //appx is the x component of apporaching vector
    vector<double> axis_x,axis_y,axis_z;
    Mat3x3 R1,R2;
    axis_x = normal_form(get_vector_3(appx,appy,appz));
    axis_y = get_vector_3(-axis_x[1],axis_x[0],0);
    if(norm(axis_y) == 0) {
        axis_y = get_vector_3(0,1,0);
    }
    else {
        axis_y = normal_form(axis_y);
    }
    axis_z = cross(axis_x,axis_y);
    R1 = Mat3x3
    (
        get_vector_3(1,0,0),
        get_vector_3(0,cos(angle),-sin(angle)),
        get_vector_3(0,sin(angle),cos(angle)),
        0
    );
    R2 = Mat3x3
    (
        axis_x,
        axis_y,
        axis_z,
        1
    );
    return R2.matmul(R1);

    /* # axis_x = towards
    # axis_y = np.array([-axis_x[1], axis_x[0], 0])
    # axis_x = axis_x / np.linalg.norm(axis_x)
    # axis_y = axis_y / np.linalg.norm(axis_y)
    # axis_z = np.cross(axis_x, axis_y)
    # R1 = np.array([[1, 0, 0],
    #             [0, np.cos(angle), -np.sin(angle)],
    #             [0, np.sin(angle), np.cos(angle)]])
    # R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
    # matrix = R2.dot(R1)
    # return matrix.astype(np.float32) */
}

double norm(vector<double> v)
{
    double sum = 0;
    for(int i=0;i < v.size();i++)
    {
        sum += pow(v[i],2.0);
    }
    return sqrt(sum);
}

vector<double> normal_form(vector<double> v)
{
    double the_norm = norm(v);
    return get_vector_3(v[0] / the_norm,v[1] / the_norm, v[2] / the_norm);
}

vector<double> cross(vector<double> v1,vector<double> v2)
{
    return get_vector_3
    (
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    );
}