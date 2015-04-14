// Here we'll store custom operators that are only used by
// certain networks. e.g spaun.

#include "custom_ops.hpp"

SpaunStimulus::SpaunStimulus(
        SignalView output, dtype* time_pointer, vector<string> stim_sequence,
        float present_interval, float present_blanks)
:output(output), time_pointer(time_pointer), previous_index(-1),
        stim_sequence(stim_sequence), present_interval(present_interval),
        present_blanks(present_blanks){

    if(stim_sequence.empty()){
        throw runtime_error("Cannot create SpaunStimulus with empty stimulus sequence.");
    }

    num_stimuli = stim_sequence.size();

    string vision_data_loc = getenv("HOME");
    vision_data_loc += "/spaun2.0/_spaun/vision/spaun_vision_data.csv";

    auto image_data = load_image_data(vision_data_loc);

    if(image_data.find("NULL") != image_data.end()){
        throw runtime_error("Found image with label NULL in data.");
    }

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

    for(string label: stim_sequence){
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
            output = SignalView(
                *images[index], ublas::slice(0, 1, image_size),
                ublas::slice(0, 1, 1));
        }

        previous_index = index;
    }
}

string SpaunStimulus::to_string() const{
    stringstream out;

    out << "SpaunStimulus:" << endl;
    out << "Num stimuli:" << num_stimuli << endl;
    out << "Present interval:" << present_interval << endl;
    out << "Present blanks:" << present_blanks << endl;
    out << "Image size:" << image_size << endl;
    out << "Previous index:" << previous_index << endl;
    out << "Stimulus sequence:" << endl;

    int i = 0;
    for(auto s: stim_sequence){
        if(i > 0){
            out << ", ";
        }

        out << s;
        i++;
    }

    out << endl;

    return out.str();

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
