#ifndef NENGO_MPI_CUSTOM_OPS_HPP
#define NENGO_MPI_CUSTOM_OPS_HPP

#include <string>
#include <vector>
#include <map>
#include <exception>
#include <fstream>
#include <cmath>
#include <time.h>

#include "operator.hpp"

using namespace std;

class SpaunStimulus: public Operator{
public:
    SpaunStimulus(Matrix output, dtype* time_pointer, vector<string> stim_seq);
    string classname() const {return "SpaunStimulus"; }

    void operator() ();

    virtual string to_string() const;

protected:
    dtype* time_pointer;

    int num_stimuli;
    int present_blanks;
    float present_interval;

    int image_size;
    vector<BaseMatrix*> images;
    BaseMatrix* null_image;
    Matrix output;

    int previous_index;
};

BaseMatrix* do_down_sample(BaseMatrix* image, int new_size);
map<string, vector<BaseMatrix*> > load_image_data(string filename);

#endif
