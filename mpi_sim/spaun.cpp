#include "spaun.hpp"

SpaunStimulus::SpaunStimulus(
    SignalView output, dtype* time_pointer, vector<string> stim_sequence,
    dtype present_interval, dtype present_blanks)
:output(output), time_pointer(time_pointer), previous_index(-1),
stim_sequence(stim_sequence), present_interval(present_interval),
present_blanks(present_blanks){

    if(stim_sequence.empty()){
        throw runtime_error("Cannot create SpaunStimulus with empty stimulus sequence.");
    }

    n_stimuli = stim_sequence.size();

    char* home = getenv("HOME");
    if(!home){
        throw runtime_error(
            "Error in creating SpaunStimulus. HOME environment variable not set.");
    }

    string vision_data_dir(home);
    vision_data_dir += "/spaun2.0/_spaun/vision/spaun_vision_data";

    image_size = output.size1();

    ImageStore image_store = ImageStore(vision_data_dir, image_size);

    int stim_count = 0;
    for(string label: stim_sequence){
        cout << "Loading image for stimulus " << stim_count << " with label " << label << endl;

        unique_ptr<BaseSignal> image;
        if(label == "None" || label == "NULL"){
            image = unique_ptr<BaseSignal>(new BaseSignal(ScalarSignal(image_size, 1, 0.0)));
        }else{
            image = image_store.get_image_with_label(label);
        }

        images.push_back(move(image));

        stim_count++;
    }
}

void SpaunStimulus::operator() (){

    dtype index_f = (*time_pointer) / present_interval / pow(2, present_blanks);

    // Need to do this because if index_f is nearly an
    // int, doesn't get converted to an int properly.
    index_f += 0.000001;
    int index = int(index_f);

    if ((present_blanks && index != int(round(index_f))) || index >= n_stimuli){
        index = n_stimuli;
    }

    if(index != previous_index){
        if(index >= n_stimuli){
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

ImageStore::ImageStore(string dir_name, int desired_img_size)
:dir_name(dir_name), desired_img_size(desired_img_size), loaded_img_size(-1){
    srand (time(NULL));
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

unique_ptr<BaseSignal> ImageStore::get_image_with_label(string label){
    if(image_counts.find(label) == image_counts.end()){
        stringstream ss;
        ss << "Image store contains no images with label " << label << ".";
        throw runtime_error(ss.str());
    }

    int index = rand() % image_counts[label];
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

    unique_ptr<BaseSignal> image = python_list_to_signal(str_data);

    if(loaded_img_size == -1){
        loaded_img_size = image->size1();
    }

    if(desired_img_size < loaded_img_size){
        image = do_down_sample(move(image), desired_img_size);
    }else if(desired_img_size > loaded_img_size){
        throw runtime_error("SpaunStimulus: loaded images too small.");
    }

    return image;
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

void print_image(BaseSignal* image){
    int s = (int) sqrt(image->size1());
    cout << "Using image dimension: " << s << endl;
    for(int i = 0; i < s; i++){
        for(int j = 0; j < s; j++){
            if(j > 0){
                cout << ", ";
            }

            if((*image)(i * s + j, 0) > 0.0){
                cout << 1;
            }else{
                cout << 0;
            }
        }
        cout << endl;
    }
}
