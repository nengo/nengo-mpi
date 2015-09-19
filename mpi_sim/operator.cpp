#include "operator.hpp"

// Constructors
Reset::Reset(SignalView dst, dtype value)
:dst(dst), value(value){

    dummy = ScalarSignal(dst.size1(), dst.size2(), value);
}

Copy::Copy(SignalView dst, SignalView src)
:dst(dst), src(src){}

DotInc::DotInc(SignalView A, SignalView X, SignalView Y)
:A(A), X(X), Y(Y){}

ElementwiseInc::ElementwiseInc(SignalView A, SignalView X, SignalView Y)
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

NoDenSynapse::NoDenSynapse(
    SignalView input, SignalView output, dtype b)
:input(input), output(output), b(b){}

SimpleSynapse::SimpleSynapse(
    SignalView input, SignalView output, dtype a, dtype b)
:input(input), output(output), a(a), b(b){}

Synapse::Synapse(
    SignalView input, SignalView output, BaseSignal numer, BaseSignal denom)
:input(input), output(output), numer(numer), denom(denom){

    for(int i = 0; i < input.size1(); i++){
        x.push_back(boost::circular_buffer<dtype>(numer.size1()));
        y.push_back(boost::circular_buffer<dtype>(denom.size1()));
    }
}

LIF::LIF(
    int n_neurons, dtype tau_rc, dtype tau_ref, dtype min_voltage,
    dtype dt, SignalView J, SignalView output, SignalView voltage,
    SignalView ref_time)
:n_neurons(n_neurons), dt(dt), tau_rc(tau_rc), tau_ref(tau_ref),
min_voltage(min_voltage), dt_inv(1.0 / dt), J(J), output(output),
voltage(voltage), ref_time(ref_time){

    one = ScalarSignal(n_neurons, 1, 1.0);
    dt_vec = ScalarSignal(n_neurons, 1, dt);
}

LIFRate::LIFRate(
    int n_neurons, dtype tau_rc, dtype tau_ref, SignalView J, SignalView output)
:n_neurons(n_neurons), tau_rc(tau_rc), tau_ref(tau_ref), J(J), output(output){}

AdaptiveLIF::AdaptiveLIF(
    int n_neurons, dtype tau_n, dtype inc_n, dtype tau_rc, dtype tau_ref,
    dtype min_voltage, dtype dt, SignalView J, SignalView output, SignalView voltage,
    SignalView ref_time, SignalView adaptation)
:LIF(n_neurons, tau_rc, tau_ref, min_voltage, dt, J, output, voltage, ref_time),
tau_n(tau_n), inc_n(inc_n), adaptation(adaptation){}

AdaptiveLIFRate::AdaptiveLIFRate(
    int n_neurons, dtype tau_n, dtype inc_n, dtype tau_rc, dtype tau_ref, dtype dt,
    SignalView J, SignalView output, SignalView adaptation)
:LIFRate(n_neurons, tau_rc, tau_ref, J, output),
tau_n(tau_n), inc_n(inc_n), dt(dt), adaptation(adaptation){}

RectifiedLinear::RectifiedLinear(int n_neurons, SignalView J, SignalView output)
:n_neurons(n_neurons), J(J), output(output){}

Sigmoid::Sigmoid(int n_neurons, dtype tau_ref, SignalView J, SignalView output)
:n_neurons(n_neurons), tau_ref(tau_ref), tau_ref_inv(1.0 / tau_ref), J(J), output(output){}

Izhikevich::Izhikevich(
    int n_neurons, dtype tau_recovery, dtype coupling, dtype reset_voltage,
    dtype reset_recovery, dtype dt, SignalView J, SignalView output,
    SignalView voltage, SignalView recovery)
:n_neurons(n_neurons), tau_recovery(tau_recovery), coupling(coupling),
reset_voltage(reset_voltage), reset_recovery(reset_recovery), dt(dt), dt_inv(1.0/dt),
J(J), output(output), voltage(voltage), recovery(recovery){

    bias = ScalarSignal(n_neurons, 1, 140);
}

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

        for(int Y_i = 0; Y_i < Y.size1(); Y_i++){
            A_j = 0;
            X_j = 0;

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

void NoDenSynapse::operator() (){
    output = b * input;
}

void SimpleSynapse::operator() (){
    output *= -a;
    output += b * input;
}

void Synapse::operator() (){
    for(int i = 0; i < input.size1(); i++){

        x[i].push_front(input(i, 0));

        output(i, 0) = 0.0;

        for(int j = 0; j < x[i].size(); j++){
            output(i, 0) += numer(j, 0) * x[i][j];
        }

        for(int j = 0; j < y[i].size(); j++){
            output(i, 0) -= denom(j, 0) * y[i][j];
        }

        y[i].push_front(output(i, 0));
    }

    run_dbg(*this);
}

void LIF::operator() (){
    dV = -expm1(-dt / tau_rc) * (J - voltage);
    voltage += dV;
    for(unsigned i = 0; i < n_neurons; ++i){
        voltage(i, 0) = voltage(i, 0) < min_voltage ? min_voltage : voltage(i, 0);
    }

    ref_time -= dt_vec;

    mult = ref_time;
    mult *= -dt_inv;
    mult += one;

    for(unsigned i = 0; i < n_neurons; ++i){
        mult(i, 0) = mult(i, 0) > 1 ? 1.0 : mult(i, 0);
        mult(i, 0) = mult(i, 0) < 0 ? 0.0 : mult(i, 0);
    }

    dtype overshoot;
    for(unsigned i = 0; i < n_neurons; ++i){
        voltage(i, 0) *= mult(i, 0);
        if(voltage(i, 0) > 1.0){
            output(i, 0) = dt_inv;
            overshoot = (voltage(i, 0) - 1.0) / dV(i, 0);
            ref_time(i, 0) = tau_ref + dt * (1.0 - overshoot);
            voltage(i, 0) = 0.0;
        }
        else
        {
            output(i, 0) = 0.0;
        }
    }

    run_dbg(*this);
}

void LIFRate::operator() (){
    for(unsigned i = 0; i < n_neurons; ++i){
        if(J(i, 0) > 1.0){
            output(i, 0) = 1.0 / (tau_ref + tau_rc * log1p(1.0 / (J(i, 0) - 1.0)));
        }else{
            output(i, 0) = 0.0;
        }
    }

    run_dbg(*this);
}

void AdaptiveLIF::operator() (){
    temp = J;
    J -= adaptation;
    LIF::operator()();
    J = temp;

    adaptation += (dt / tau_n) * (inc_n * output - adaptation);

    run_dbg(*this);
}

void AdaptiveLIFRate::operator() (){
    temp = J;
    J -= adaptation;
    LIFRate::operator()();
    J = temp;

    adaptation += (dt / tau_n) * (inc_n * output - adaptation);

    run_dbg(*this);
}

void RectifiedLinear::operator() (){
    dtype j = 0;
    for(unsigned i = 0; i < n_neurons; ++i){
        j = J(i, 0);
        output(i, 0) = j > 0.0 ? j : 0.0;
    }

    run_dbg(*this);
}

void Sigmoid::operator() (){
    for(unsigned i = 0; i < n_neurons; ++i){
        output(i, 0) = tau_ref_inv / (1.0 + exp(-J(i, 0)));
    }

    run_dbg(*this);
}

void Izhikevich::operator() (){
    for(unsigned i = 0; i < n_neurons; ++i){
        J(i, 0) = J(i, 0) > -30 ? J(i, 0) : -30;
    }

    voltage_squared = 0.04 * element_prod(voltage, voltage);

    dV = 5 * voltage;
    dV += voltage_squared + bias + J - recovery;
    dV *= 1000 * dt;
    voltage += dV;

    for(unsigned i = 0; i < n_neurons; ++i){
        if(voltage(i, 0) >= 30){
            output(i, 0) = dt_inv;
            voltage(i, 0) = reset_voltage;
        }else{
            output(i, 0) = 0.0;
        }
    }

    dU = coupling * voltage;
    dU -= recovery;
    dU *= tau_recovery * 1000 * dt;
    recovery += dU;

    for(unsigned i = 0; i < n_neurons; ++i){
        if(output(i, 0) > 0){
            recovery(i, 0) += reset_recovery;
        }
    }

    run_dbg(*this);
}

string signal_to_string(const SignalView signal) {

    stringstream ss;

    if(RUN_DEBUG_TEST){
        ss << signal;
    }else{
        ss << "[" << signal.size1() << ", " << signal.size2() << "]";
    }

    return ss.str();
}

//to_string
string Reset::to_string() const {

    stringstream out;
    out << Operator::to_string();
    out << "dst:" << endl;
    out << signal_to_string(dst) << endl;

    return out.str();
}

string Copy::to_string() const  {

    stringstream out;
    out << Operator::to_string();
    out << "dst:" << endl;
    out << signal_to_string(dst) << endl;
    out << "src:" << endl;
    out << signal_to_string(src) << endl;

    return out.str();
}

string DotInc::to_string() const{

    stringstream out;
    out << Operator::to_string();
    out << "A:" << endl;
    out << signal_to_string(A) << endl;
    out << "X:" << endl;
    out << signal_to_string(X) << endl;
    out << "Y:" << endl;
    out << signal_to_string(Y) << endl;

    return out.str();
}

string ElementwiseInc::to_string() const{

    stringstream out;
    out << Operator::to_string();
    out << "A:" << endl;
    out << signal_to_string(A) << endl;
    out << "X:" << endl;
    out << signal_to_string(X) << endl;
    out << "Y:" << endl;
    out << signal_to_string(Y) << endl;

    out << "Broadcast: " << broadcast << endl;
    out << "A_row_stride: " << A_row_stride << endl;
    out << "A_col_stride: " << A_col_stride << endl;

    out << "X_row_stride: " << X_row_stride << endl;
    out << "X_col_stride: " << X_col_stride << endl;

    return out.str();
}

string NoDenSynapse::to_string() const{

    stringstream out;
    out << Operator::to_string();
    out << "input:" << endl;
    out << signal_to_string(input) << endl;
    out << "output:" << endl;
    out << signal_to_string(output) << endl;
    out << "b: " << b << endl;

    return out.str();
}


string SimpleSynapse::to_string() const{

    stringstream out;
    out << Operator::to_string();
    out << "input:" << endl;
    out << signal_to_string(input) << endl;
    out << "output:" << endl;
    out << signal_to_string(output) << endl;
    out << "a: " << a << endl;
    out << "b: " << b << endl;

    return out.str();
}

string Synapse::to_string() const{

    stringstream out;
    out << Operator::to_string();
    out << "input:" << endl;
    out << signal_to_string(input) << endl;
    out << "output:" << endl;
    out << signal_to_string(output) << endl;
    out << "numer:" << endl;
    out << numer << endl;
    out << "denom:" << endl;
    out << denom << endl;

    /*
    out << "x & y:" << endl;
    for(int i = 0; i < input.size(); i++){
        out << "i: " << i << endl;

        out << "x.size " << x[i].size() << endl;
        for(int j = 0; j < x[i].size(); j++){
            out << "x[ "<< j << "] = "<< x[i][j] << ", ";
        }
        out << endl;

        out << "y.size " << y[i].size() << endl;
        for(int j = 0; j < y[i].size(); j++){
            out << "y[ "<< j << "] = "<< y[i][j] << ", ";
        }
        out << endl;
    }
    */

    return out.str();
}

string LIF::to_string() const{

    stringstream out;

    out << Operator::to_string();
    out << "J:" << endl;
    out << signal_to_string(J) << endl;
    out << "output:" << endl;
    out << signal_to_string(output) << endl;
    out << "voltage:" << endl;
    out << signal_to_string(voltage) << endl;
    out << "refractory_time:" << endl;
    out << signal_to_string(ref_time) << endl;
    out << "n_neurons: " << n_neurons << endl;;
    out << "tau_rc: " << tau_rc << endl;
    out << "tau_ref: " << tau_ref << endl;
    out << "min_voltage: " << min_voltage << endl;

    return out.str();
}

string LIFRate::to_string() const{

    stringstream out;
    out << Operator::to_string();
    out << "J:" << endl;
    out << signal_to_string(J) << endl;
    out << "output:" << endl;
    out << signal_to_string(output) << endl;
    out << "n_neurons: " << n_neurons << endl;
    out << "tau_rc: " << tau_rc << endl;
    out << "tau_ref: " << tau_ref << endl;

    return out.str();
}

string AdaptiveLIF::to_string() const{

    stringstream out;
    out << LIF::to_string();
    out << "tau_n: " << tau_n << endl;
    out << "inc_n: " << inc_n << endl;
    out << "adaptation: " << endl;
    out << signal_to_string(adaptation) << endl;

    return out.str();
}

string AdaptiveLIFRate::to_string() const{

    stringstream out;
    out << LIFRate::to_string();
    out << "tau_n: " << tau_n << endl;
    out << "inc_n: " << inc_n << endl;
    out << "dt: " << dt << endl;
    out << "adaptation: " << endl;
    out << signal_to_string(adaptation) << endl;

    return out.str();
}

string RectifiedLinear::to_string() const{

    stringstream out;
    out << Operator::to_string();
    out << "n_neurons: " << n_neurons << endl;
    out << "J:" << endl;
    out << signal_to_string(J) << endl;
    out << "output:" << endl;
    out << signal_to_string(output) << endl;

    return out.str();
}

string Sigmoid::to_string() const{

    stringstream out;
    out << Operator::to_string();
    out << "n_neurons: " << n_neurons << endl;
    out << "tau_ref: " << tau_ref << endl;
    out << "J:" << endl;
    out << signal_to_string(J) << endl;
    out << "output:" << endl;
    out << signal_to_string(output) << endl;

    return out.str();
}

string Izhikevich::to_string() const{

    stringstream out;
    out << Operator::to_string();
    out << "n_neurons: " << n_neurons << endl;
    out << "tau_recovery: " << tau_recovery << endl;
    out << "coupling: " << coupling << endl;
    out << "reset_voltage: " << reset_voltage << endl;
    out << "reset_recovery: " << reset_recovery << endl;
    out << "dt: " << dt << endl;

    out << "J:" << endl;
    out << signal_to_string(J) << endl;
    out << "output:" << endl;
    out << signal_to_string(output) << endl;
    out << "voltage:" << endl;
    out << signal_to_string(voltage) << endl;
    out << "recovery:" << endl;
    out << signal_to_string(recovery) << endl;

    return out.str();
}

unique_ptr<BaseSignal> extract_float_list(string s){
    // Remove surrounding square brackets
    boost::trim_if(s, boost::is_any_of("[]"));

    vector<string> tokens;
    boost::split(tokens, s, boost::is_any_of(","));

    unique_ptr<BaseSignal> result(new BaseSignal(tokens.size(), 1));

    try{
        int i = 0;
        for(string token: tokens){
            boost::trim(token);
            (*result)(i, 0) = boost::lexical_cast<dtype>(token);
            i++;
        }
    }catch(const boost::bad_lexical_cast& e){
        cout << "Caught bad lexical cast while extracting list "
                "with error " << e.what() << endl;
        terminate();
    }

    return result;
}
