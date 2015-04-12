// Here we'll store custom operators that are only used by
// certain networks. e.g spaun.

#include "custom_ops.hpp"

SpaunStimulus::SpaunStimulus(SignalView output, dtype* time_pointer, vector<string> stim_seq)
:output(output), time_pointer(time_pointer), previous_index(-1){

    if(stim_seq.empty()){
        stim_seq = {
            "A", "ONE", "OPEN", "1", "2", "3", "4", "5", "1", "2", "CLOSE", "QM"};
    }

    num_stimuli = stim_seq.size();
    present_interval = 1.0;
    present_blanks = 1.0;

    int num_mtr_responses = 7;
    float mtr_ramp_scale = 2.0;
    float mtr_est_digit_response_time = 1.5 / mtr_ramp_scale;

    int est_mtr_response_time = num_mtr_responses * mtr_est_digit_response_time;
    float est_run_time = stim_seq.size() * present_interval * pow(2, present_blanks);
    int extra_spaces = int(est_mtr_response_time / (present_interval * pow(2, present_blanks)));

    for(int i = 0; i < extra_spaces; i++){
        stim_seq.push_back("");
    }

    string vision_data_loc = getenv("HOME");
    vision_data_loc += "/spaun2.0/_spaun/vision/spaun_vision_data.csv";

    auto image_data = load_image_data(vision_data_loc);

    int loaded_image_size = image_data.at("0")[0]->size1();
    image_size = output.size1();

    bool down_sample = false;

    if(image_size < loaded_image_size){
        cout << "SpaunStimulus: loaded images have " << loaded_image_size
             << " dimensions, but network requires images with dimension "
             << image_size << ". Downsampling." << endl;
        down_sample = true;

    }else if(image_size > loaded_image_size){
        throw runtime_error("SpaunStimulus: loaded images too small.");
    }

    map<string, string> debug_mapping = {
        {"A", "SIX"}, {"OPEN", "SEV"}, {"CLOSE", "EIG"}, {"QM", "NIN"}};

    srand (time(NULL));

    for(string label: stim_seq){
        if(debug_mapping.find(label) != debug_mapping.end()){
            label = debug_mapping.at(label);
        }

        unique_ptr<BaseSignal> image;

        if(!label.empty() && image_data.find(label) != image_data.end()){
            // Pick a random image with the specified label
            vector<unique_ptr<BaseSignal>>& label_images = image_data.at(label);

            int index = rand() % label_images.size();

            image = unique_ptr<BaseSignal>(new BaseSignal(*label_images.at(index)));

            if(down_sample){
                image = do_down_sample(move(image), image_size);
            }
        }else{
            image = unique_ptr<BaseSignal>(new BaseSignal(ScalarSignal(image_size, 1, 0.0)));
        }

        cout << "Image for label: " << label << endl;
        cout << *image << endl;

        images.push_back(move(image));
    }
}

void SpaunStimulus::operator() (){
    float index_f = (*time_pointer) / present_interval / pow(2, present_blanks);
    int index = int(index_f);

    if (present_blanks &&
            (index != int(round(index_f)) || index >= num_stimuli)){
        index = num_stimuli;
    }

    if(index != previous_index){
        if(index == num_stimuli){
            output = ScalarSignal(image_size, 1, 0.0);
        }else{
            // TODO: Should have a check somewhere to make sure output is
            // the same size as the images that we've retrieved
            output = SignalView(
                *images[index], ublas::slice(0, 1, image_size),
                ublas::slice(0, 1, 1));
        }

        previous_index = index;
    }
}

string SpaunStimulus::to_string() const{

}

map<string, vector<unique_ptr<BaseSignal>>> load_image_data(string filename){

    ifstream ifs(filename);

    if(!ifs.good()){
        stringstream s;
        s << "Could not load image data from " << filename << ". File does not exist." << endl;
        throw runtime_error(s.str());
    }

    map<string, vector<unique_ptr<BaseSignal>>> image_map;
    string label, data;

    while(ifs.good()){
        getline(ifs, label);
        boost::replace_all(label, ",", "");

        getline(ifs, data);

        unique_ptr<BaseSignal> image = extract_float_list(data);

        image_map[label].push_back(move(image));
    }

    ifs.close();

    return image_map;
}

unique_ptr<BaseSignal> do_down_sample(unique_ptr<BaseSignal> image, int new_size){
    auto new_image = unique_ptr<BaseSignal>(new BaseSignal(new_size, 1));

    // For now, just randomly choose the pixels for the new image from the old image
    srand (time(NULL));

    vector<int> chosen_indices;

    while(chosen_indices.size() < new_size){
        int index = rand() % image->size1();

        auto iter = find(chosen_indices.begin(), chosen_indices.end(), index);
        if(iter == chosen_indices.end()){
            chosen_indices.push_back(index);
        }
    }

    sort(chosen_indices.begin(), chosen_indices.end());

    int new_index = 0;

    for(auto index : chosen_indices){
        (*new_image)(new_index, 0) = (*image)(index, 0);
        new_index++;
    }

    return new_image;
}
