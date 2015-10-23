#ifndef NENGO_MPI_CUSTOM_OPS_HPP
#define NENGO_MPI_CUSTOM_OPS_HPP

#include <string>
#include <vector>
#include <map>
#include <exception>
#include <fstream>
#include <sstream>
#include <memory>
#include <cmath>
#include <time.h>

#include "operator.hpp"
#include "utils.hpp"

using namespace std;

class SpaunStimulus: public Operator{
public:
    SpaunStimulus(
        SignalView output, dtype* time_pointer, vector<string> stim_sequence,
        dtype present_interval, dtype present_blanks);

    string classname() const {return "SpaunStimulus"; }

    void operator() ();
    virtual string to_string() const;

    virtual void reset(unsigned seed);

protected:
    dtype* time_pointer;

    int n_stimuli;
    string vision_data_dir;
    vector<string> stim_sequence;
    dtype present_interval;
    dtype present_blanks;

    int image_size;
    vector<unique_ptr<BaseSignal>> images;
    SignalView output;

    int previous_index;
};

class ImageStore{
public:
    ImageStore(string dir_name, int desired_img_size, unsigned seed);

    void load_image_counts(string filename);

    // Get a random image with the given label
    unique_ptr<BaseSignal> get_image_with_label(string label);

protected:
    string dir_name;
    map<string, int> image_counts;

    int desired_img_size;

    // -1 initially; set properly when we load the first image
    int loaded_img_size;

    default_random_engine rng;
};

/*
 * Down-sample the given image, returning an image whose size is new_size.
 * new_size should be < the size of the given image. Currently downsamples
 * by randomly choosing indices without replacement and then sorting. */
unique_ptr<BaseSignal> do_down_sample(unique_ptr<BaseSignal> image, int new_size);

void print_image(BaseSignal* image);

#endif
