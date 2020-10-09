// Code Written by Minghao Gou

#include "graspnms.h"
#define SCORE_PLACE 0
#define TRANSLATION_START_PLACE 13
#define ROTATION_START_PLACE 4

using namespace std;

tuple_thresh::tuple_thresh()
{
    this->translation_thresh = 0;
    this->rotation_thresh = 0;
}

tuple_thresh::tuple_thresh(double t, double r)
{
    this->translation_thresh = t;
    this->rotation_thresh = r;
}

int tuple_thresh::smaller(tuple_thresh other)
{
    return this->translation_thresh < other.translation_thresh && this->rotation_thresh < other.rotation_thresh;
}

void tuple_thresh::print_thresh()
{
    cout << "tuple_thresh: t=" << this->translation_thresh << " r=" << this->rotation_thresh << endl;
}

double_array::double_array()
{
    this->r = 0;
    this->c = 0;
    this->data = NULL;
}

double_array::double_array(int r, int c)
{
    this->r = r;
    this->c = c;
    this->data = new double[r * c];
}

double_array::~double_array()
{
    this->data = NULL;
    this->r = 0;
    this->c = 0;
}

void double_array::print_data()
{
    cout << "print data" << endl;
    for (int i = 0; i < this->r; i++)
    {
        for (int j = 0; j < this->c; j++)
        {
            printf("%.4f\t", (this->data)[i * (this->c) + j]);
        }
        cout << endl;
    }
    // argsort_grasp(*this);
}

double_array grasp_nms(double_array da_in, tuple_thresh th)
{
    vector<unsigned int> sorted_arg_v = argsort_grasp(da_in);
    list<unsigned int> sorted_arg(sorted_arg_v.begin(), sorted_arg_v.end());
    list<unsigned int>::iterator current_num_grasp = sorted_arg.begin();
    list<unsigned int>::iterator num_grasp, next_grasp;
    while (current_num_grasp != sorted_arg.end())
    {
        num_grasp = current_num_grasp;
        num_grasp++;
        while (num_grasp != sorted_arg.end())
        {
            if (iou(da_in.data + da_in.c * (*current_num_grasp), da_in.data + da_in.c * (*num_grasp)).smaller(th))
            {
                num_grasp = sorted_arg.erase(num_grasp);
            }
            else
            {
                num_grasp++;
            }
        }
        current_num_grasp++;
    }
    double_array da_out(sorted_arg.size(), da_in.c);
    num_grasp = sorted_arg.begin();
    for (int i = 0; i < sorted_arg.size(); i++)
    {
        memcpy(da_out.data + i * da_in.c, da_in.data + (*num_grasp) * da_in.c, da_in.c * sizeof(double));
        num_grasp++;
    }
    return da_out;
}

bool array_gt(arg_value i1, arg_value i2)
{
    return i1.score > i2.score;
}

template <typename T>
std::vector<unsigned int> reverse_argsort(const std::vector<T> &array)
{
    const int array_len(array.size());
    std::vector<arg_value> array_index;
    array_index.reserve(array_len);

    for (int i = 0; i < array_len; ++i)
    {
        arg_value arg_value_struct;
        arg_value_struct.arg = i;
        arg_value_struct.score = array[i];
        array_index.push_back(arg_value_struct);
    }
    std::sort(array_index.begin(), array_index.end(), array_gt);
    std::vector<unsigned int> index;
    index.reserve(array_len);
    for (int i = 0; i < array_len; i++)
    {
        index.push_back(array_index[i].arg);
    }
    return index;
}

vector<unsigned int> argsort_grasp(const double_array &grasps)
{
    vector<double> grasp_scores;
    vector<unsigned int> sorted_arg;
    grasp_scores.reserve(grasps.r);
    for (int i = 0; i < grasps.r; i++)
    {
        grasp_scores.push_back(grasps.data[i * grasps.c + SCORE_PLACE]);
    }
    sorted_arg = reverse_argsort(grasp_scores);
    return sorted_arg;
}

tuple_thresh iou(double *pg1, double *pg2)
{
    // 0: score
    // 1: width
    // 2: height
    // 3: depth
    // 4-12: rotation
    // 13-15: translation
    // 16: object id
    double sum_of_square = 0;
    for (int i = TRANSLATION_START_PLACE; i < TRANSLATION_START_PLACE + 3; i++)
    {
        sum_of_square += pow(
            *(pg1 + i) - *(pg2 + i),
            2.0);
    }
    double t = sqrt(sum_of_square);
    // translation
    Mat3x3 m1 = rotation_array_to_matrix(pg1 + ROTATION_START_PLACE);
    Mat3x3 m2 = rotation_array_to_matrix(pg2 + ROTATION_START_PLACE);
    // Mat3x3 m1 = viewpoint_params_to_matrix(*(pg1 + 5), *(pg1 + 6), *(pg1 + 7), *(pg1 + 8));
    // Mat3x3 m2 = viewpoint_params_to_matrix(*(pg2 + 5), *(pg2 + 6), *(pg2 + 7), *(pg2 + 8));
    double trace = m1.matmul(m2.T()).trace();
    trace = (trace > 3.0) ? 3.0 : trace;
    trace = (trace < -1.0) ? -1.0 : trace;
    double r = acos(0.5 * (trace - 1.0));
    // rotation
    return tuple_thresh(t, r);
}
