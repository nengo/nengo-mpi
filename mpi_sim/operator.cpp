#include "operator.hpp"

// Constructors
Reset::Reset(Matrix dst, dtype value)
:dst(dst), value(value){

    dummy = ScalarMatrix(dst.size1(), dst.size2(), value);
}

Copy::Copy(Matrix dst, Matrix src)
:dst(dst), src(src){}

DotInc::DotInc(Matrix A, Matrix X, Matrix Y)
:A(A), X(X), Y(Y){}

ElementwiseInc::ElementwiseInc(Matrix A, Matrix X, Matrix Y)
:A(A), X(X), Y(Y){

    if(A.size1() != Y.size1() || A.size2() != Y.size2() ||
       X.size1() != Y.size1() || X.size2() != Y.size2()){
        broadcast = true;
        A_row_stride = A.size1() > 1 ? 1 : 0;
        A_col_stride = A.size2() > 1 ? 1 : 0;

        X_row_stride = X.size1() > 1 ? 1 : 0;
        X_col_stride = X.size2() > 1 ? 1 : 0;
    }else{
        broadcast = false;
        A_row_stride = 1;
        A_col_stride = 1;

        X_row_stride = 1;
        X_col_stride = 1;

    }
}

Synapse::Synapse(
    Matrix input, Matrix output, BaseMatrix* numer, BaseMatrix* denom)
:input(input), output(output), numer(numer), denom(denom){

    for(int i = 0; i < input.size1(); i++){
        x.push_back(boost::circular_buffer<dtype>(numer->size1()));
        y.push_back(boost::circular_buffer<dtype>(denom->size1()));
    }
}

SimLIF::SimLIF(
    int n_neurons, dtype tau_rc, dtype tau_ref,
    dtype dt, Matrix J, Matrix output)
:n_neurons(n_neurons), dt(dt), tau_rc(tau_rc), tau_ref(tau_ref),
dt_inv(1.0 / dt), J(J), output(output){

    voltage = BaseMatrix(n_neurons, 1, 0.0);
    refractory_time = BaseMatrix(n_neurons, 1, 0.0);
    one = ScalarMatrix(n_neurons, 1, 1.0);
    dt_vec = ScalarMatrix(n_neurons, 1, dt);
}

SimLIFRate::SimLIFRate(
    int n_neurons, dtype tau_rc,
    dtype tau_ref, Matrix J, Matrix output)
:n_neurons(n_neurons), tau_rc(tau_rc), tau_ref(tau_ref), J(J), output(output){

    j = BaseMatrix(n_neurons, 1);
    one = ScalarMatrix(n_neurons, 1, 1.0);
}

SimRectifiedLinear::SimRectifiedLinear(int n_neurons, Matrix J, Matrix output)
:n_neurons(n_neurons), J(J), output(output){}

SimSigmoid::SimSigmoid(int n_neurons, dtype tau_ref, Matrix J, Matrix output)
:n_neurons(n_neurons), tau_ref(tau_ref), tau_ref_inv(1.0 / tau_ref), J(J), output(output){}

// Function operator overloads
void Reset::operator() (){

    dst = dummy;

    run_dbg(*this);
}

void Copy::operator() (){

    dst = src;

    run_dbg(*this);
}

void DotInc::operator() (){
    axpy_prod(A, X, Y, false);

    run_dbg(*this);
}

void ElementwiseInc::operator() (){
    if(broadcast){
        int A_i = 0, A_j = 0, X_i = 0, X_j = 0;

        for(int Y_i = 0;Y_i < Y.size1(); Y_i++){
            for(int Y_j = 0;Y_j < Y.size2(); Y_j++){
                Y(Y_i, Y_j) += A(A_i, A_j) * X(X_i, X_j);
                A_j += A_col_stride;
                X_j += X_col_stride;
            }

            A_i += A_row_stride;
            X_i += X_row_stride;
        }

    }else{
        Y += element_prod(A, X);
    }

    run_dbg(*this);
}

void Synapse::operator() (){
    for(int i = 0; i < input.size1(); i++){

        x[i].push_front(input(i, 0));

        output(i, 0) = 0.0;

        for(int j = 0; j < x[i].size(); j++){
            output(i, 0) += (*numer)(j, 0) * x[i][j];
        }

        for(int j = 0; j < y[i].size(); j++){
            output(i, 0) -= (*denom)(j, 0) * y[i][j];
        }

        y[i].push_front(output(i, 0));
    }

    run_dbg(*this);
}

void SimLIF::operator() (){
    dV = -expm1(-dt / tau_rc) * (J - voltage);
    voltage += dV;
    for(unsigned i = 0; i < n_neurons; ++i){
        voltage(i, 0) = voltage(i, 0) < 0 ? 0.0 : voltage(i, 0);
    }

    refractory_time -= dt_vec;

    mult = (one - refractory_time * dt_inv);

    for(unsigned i = 0; i < n_neurons; ++i){
        mult(i, 0) = mult(i, 0) > 1 ? 1.0 : mult(i, 0);
        mult(i, 0) = mult(i, 0) < 0 ? 0.0 : mult(i, 0);
    }

    dtype overshoot;
    for(unsigned i = 0; i < n_neurons; ++i){
        voltage(i, 0) *= mult(i, 0);
        if (voltage(i, 0) > 1.0){
            output(i, 0) = dt_inv;
            overshoot = (voltage(i, 0) - 1.0) / dV(i, 0);
            refractory_time(i, 0) = tau_ref + dt * (1.0 - overshoot);
            voltage(i, 0) = 0.0;
        }
        else
        {
            output(i, 0) = 0.0;
        }
    }

    run_dbg(*this);
}

void SimLIFRate::operator() (){

    j = J - one;

    for(unsigned i = 0; i < n_neurons; ++i){
        if(j(i, 0) > 0.0){
            output(i, 0) = 1.0 / (tau_ref + tau_rc * log1p(1.0 / j(i, 0)));
        }else{
            output(i, 0) = 0.0;
        }
    }

    run_dbg(*this);
}

void SimRectifiedLinear::operator() (){
    dtype j = 0;
    for(unsigned i = 0; i < n_neurons; ++i){
        j = J(i, 0);
        output(i, 0) = j > 0.0 ? j : 0.0;
    }

    run_dbg(*this);
}

void SimSigmoid::operator() (){
    for(unsigned i = 0; i < n_neurons; ++i){
        output(i, 0) = tau_ref_inv / (1.0 + exp(-J(i, 0)));
    }

    run_dbg(*this);
}

//to_string
string Reset::to_string() const {
    stringstream out;

    out << "Reset:" << endl;
    out << "dst:" << endl;
    out << dst << endl;

    return out.str();
}

string Copy::to_string() const  {
    stringstream out;

    out << "Copy:" << endl;
    out << "dst:" << endl;
    out << dst << endl;
    out << "src:" << endl;
    out << src << endl;

    return out.str();
}

string DotInc::to_string() const{
    stringstream out;

    out << "DotInc:" << endl;
    out << "A:" << endl;
    out << A << endl;
    out << "X:" << endl;
    out << X << endl;
    out << "Y:" << endl;
    out << Y << endl;

    return out.str();
}

string ElementwiseInc::to_string() const{
    stringstream out;

    out << "ElementwiseInc:" << endl;
    out << "A:" << endl;
    out << A << endl;
    out << "X:" << endl;
    out << X << endl;
    out << "Y:" << endl;
    out << Y << endl;

    out << "Broadcast: " << broadcast << endl;
    out << "A_row_stride: " << A_row_stride << endl;
    out << "A_col_stride: " << A_col_stride << endl;

    out << "X_row_stride: " << X_row_stride << endl;
    out << "X_col_stride: " << X_col_stride << endl;

    return out.str();
}

string Synapse::to_string() const{
    stringstream out;

    out << "Synapse:" << endl;

    out << "input:" << endl;
    out << input << endl;
    out << "output:" << endl;
    out << output << endl;
    out << "numer:" << endl;
    out << *numer << endl;
    out << "denom:" << endl;
    out << *denom << endl;

    /*
    out << "x & y:" << endl;
    for(int i = 0; i < input.size(); i++){
        out << "i: " << i << endl;

        out << "x.size " << x[i].size() << endl;
        for(int j = 0; j < x[i].size(); j++){
            cout << "x[ "<< j << "] = "<< x[i][j] << ", ";
        }
        out << endl;

        out << "y.size " << y[i].size() << endl;
        for(int j = 0; j < y[i].size(); j++){
            cout << "y[ "<< j << "] = "<< y[i][j] << ", ";
        }
        out << endl;
    }
    */

    return out.str();
}

string SimLIF::to_string() const{
    stringstream out;

    out << "SimLIF:" << endl;
    out << "J:" << endl;
    out << J << endl;
    out << "output:" << endl;
    out << output << endl;
    out << "voltage:" << endl;
    out << voltage << endl << endl;
    out << "refractory_time:" << endl;
    out << refractory_time << endl;

    return out.str();
}

string SimLIFRate::to_string() const{
    stringstream out;

    out << "SimLIFRate:" << endl;
    out << "J:" << endl;
    out << J << endl;
    out << "output:" << endl;
    out << output << endl;

    return out.str();
}

string SimRectifiedLinear::to_string() const{
    stringstream out;

    out << "SimRectifiedLinear:" << endl;
    out << "n_neurons: " << n_neurons << endl;
    out << "J:" << endl;
    out << J << endl;
    out << "output:" << endl;
    out << output << endl;

    return out.str();
}

string SimSigmoid::to_string() const{
    stringstream out;

    out << "SimSigmoid:" << endl;
    out << "n_neurons: " << n_neurons << endl;
    out << "tau_ref: " << tau_ref << endl;
    out << "J:" << endl;
    out << J << endl;
    out << "output:" << endl;
    out << output << endl;

    return out.str();
}

BaseMatrix* extract_float_list(string s){
    // Remove surrounding square brackets
    boost::trim_if(s, boost::is_any_of("[]"));

    vector<string> tokens;
    boost::split(tokens, s, boost::is_any_of(","));

    int length = tokens.size();
    int i = 0;
    vector<string>::const_iterator it;

    BaseMatrix* ret = new BaseMatrix(length, 1);
    try{
        for(it = tokens.begin(); it != tokens.end(); ++it){
            string val = *it;
            boost::trim(val);
            (*ret)(i, 0) = boost::lexical_cast<dtype>(val);
            i++;
        }
    }catch(const boost::bad_lexical_cast& e){
        cout << "Caught bad lexical cast while extracting list "
                "with error " << e.what() << endl;
        terminate();
    }

    return ret;
}
