#include "utils.hpp"

Signal python_list_to_signal(string s, bool get_size){
    boost::trim_if(s, boost::is_any_of("[]"));

    vector<string> tokens;
    boost::split(tokens, s, boost::is_any_of(","));

    unsigned size1, size2, offset;
    if(get_size){
        size1 = boost::lexical_cast<int>(tokens[0]);
        size2 = boost::lexical_cast<int>(tokens[1]);
        offset = 2;
    }else{
        size1 = tokens.size();
        size2 = 1;
        offset = 0;
    }

    Signal result(size1, size2);
    try{
        int i = 0;
        for(auto token = tokens.begin()+offset; token != tokens.end(); token++){
            result(int(i / size2), i % size2) = boost::lexical_cast<dtype>(*token);
            i++;
        }
    }catch(const boost::bad_lexical_cast& e){
        cout << "Caught bad lexical cast converting list to signal "
                "with error: " << e.what() << endl;
        terminate();
    }

    return result;
}

vector<int> python_list_to_index_vector(string s){
    // Remove surrounding square brackets
    boost::trim_if(s, boost::is_any_of("[]"));

    vector<string> tokens;
    boost::split(tokens, s, boost::is_any_of(","));

    vector<int> result;

    if(s.length() > 0){
        try{
            for(string token: tokens){
                boost::trim(token);
                result.push_back(boost::lexical_cast<int>(token));
            }
        }catch(const boost::bad_lexical_cast& e){
            cout << "Caught bad lexical cast while converting list "
                    "with error: " << e.what() << endl;
            terminate();
        }
    }

    return result;
}
