#include "spaun.hpp"

unique_ptr<ImageStore> SpaunStimulus::image_store = nullptr;

SpaunStimulus::SpaunStimulus(
    Signal output, Signal t, vector<string> stim_sequence,
    dtype present_interval, dtype present_blanks, int identifier)
:output(output), t(t), previous_index(-1),
stim_sequence(stim_sequence), present_interval(present_interval),
present_blanks(present_blanks), identifier(identifier){

    if(stim_sequence.empty()){
        throw runtime_error("Cannot create SpaunStimulus with empty stimulus sequence.");
    }

    n_stimuli = stim_sequence.size();

    if(image_store == nullptr){
        char* home = getenv("HOME");
        if(!home){
            throw runtime_error(
                "Error in creating SpaunStimulus. HOME environment variable not set.");
        }

        vision_data_dir = home;
        vision_data_dir += "/spaun2.0/_spaun/vision/spaun_vision_data";
        image_store = static_cast<unique_ptr<ImageStore>>(new ImageStore(vision_data_dir));
    }

    image_size = output.shape1;
}

void SpaunStimulus::operator() (){

    dtype index_f = t(0) / present_interval / pow(2, present_blanks);

    // Need to do this because if index_f is nearly an
    // int, doesn't get converted to an int properly.
    index_f += 0.000001;
    int index = int(index_f);

    if ((present_blanks && index != int(round(index_f))) || index >= n_stimuli){
        index = n_stimuli;
    }

    if(index != previous_index){
        if(index >= n_stimuli){
            dtype init_value = 0.0;
            output.fill_with(0.0);
        }else{
            output.fill_with(images[index]);
        }

        previous_index = index;
    }
}

string SpaunStimulus::to_string() const{
    stringstream out;

    out << "SpaunStimulus:" << endl;
    out << "Num stimuli:" << n_stimuli << endl;
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

void SpaunStimulus::reset(unsigned seed){
    default_random_engine rng(seed);

    // We shouldn't need to do this, but in practice I've found the first number is
    // consistently 0. Not sure why.
    rng.discard(1);

    images.clear();

    int stim_count = 0;
    for(string label: stim_sequence){
        cout << "Loading image for stimulus " << stim_count << " with label " << label << endl;

        Signal image;
        if(label == "None" || label == "NULL"){
            dtype init_value = 0.0;
            image = Signal(image_size, init_value);
        }else{
            image = image_store->get_image_with_label(label, image_size, rng);
        }

        images.push_back(image);

        stim_count++;
    }

    previous_index = -1;
}

ImageStore::ImageStore(string dir_name)
:dir_name(dir_name), loaded_img_size(-1){
    load_image_counts(dir_name + "/counts");
}

void ImageStore::load_image_counts(string filename){
    ifstream ifs(filename);

    if(!ifs.good()){
        stringstream s;
        s << "Could not load image counts from " << filename << ". File does not exist." << endl;
        throw runtime_error(s.str());
    }

    string str_label, str_n_images;
    while(ifs.good()){
        getline(ifs, str_label);
        getline(ifs, str_n_images);
        image_counts[str_label] = boost::lexical_cast<int>(str_n_images);
    }

    ifs.close();
}

Signal ImageStore::get_image_with_label(
        string label, unsigned desired_img_size, default_random_engine rng){

    if(image_counts.find(label) == image_counts.end()){
        stringstream ss;
        ss << "Image store contains no images with label " << label << ".";
        throw runtime_error(ss.str());
    }

    cout << "Image count for label " << label << ": "<< image_counts[label] << endl;
    uniform_int_distribution<int> dist(0, image_counts[label]-1);
    int index = dist(rng);

    stringstream image_file;
    image_file << dir_name << "/" << label << "/" << index;
    cout << "Loading image from file: " << image_file.str() << endl;

    ifstream ifs(image_file.str());

    if(!ifs.good()){
        stringstream s;
        s << "Could not load image from " << image_file.str() << ". File does not exist." << endl;
        throw runtime_error(s.str());
    }

    string str_data;
    getline(ifs, str_data);
    ifs.close();

    bool get_size = false;
    Signal image = python_list_to_signal(str_data, get_size);

    if(loaded_img_size == -1){
        loaded_img_size = image.shape1;
    }

    if(desired_img_size < loaded_img_size){
        image = do_down_sample(image, desired_img_size);
    }else if(desired_img_size > loaded_img_size){
        throw runtime_error("SpaunStimulus: loaded images too small.");
    }

    return image;
}

Signal do_down_sample(Signal image, unsigned new_size){
    dtype init_value = 0.0;
    Signal new_image(new_size, init_value);

    // For now, just randomly choose the pixels for the new image from the old image
    srand (time(NULL));

    vector<unsigned> chosen_indices;

    while(chosen_indices.size() < new_size){
        unsigned index = rand() % image.shape1;

        auto iter = find(chosen_indices.begin(), chosen_indices.end(), index);
        if(iter == chosen_indices.end()){
            chosen_indices.push_back(index);
        }
    }

    sort(chosen_indices.begin(), chosen_indices.end());

    unsigned new_index = 0;
    for(auto index : chosen_indices){
        new_image(new_index) = image(index);
        new_index++;
    }

    return new_image;
}

void print_image(Signal image){
    int s = (int) sqrt(image.shape1);
    cout << "Using image dimension: " << s << endl;
    for(int i = 0; i < s; i++){
        for(int j = 0; j < s; j++){
            if(j > 0){
                cout << ", ";
            }

            if(image(i * s + j) > 0.0){
                cout << 1;
            }else{
                cout << 0;
            }
        }
        cout << endl;
    }
}
