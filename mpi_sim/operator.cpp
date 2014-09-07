#include <iostream>
#include <boost/numeric/ublas/operation.hpp>

#include "operator.hpp"

// Constructors

Reset::Reset(Vector* dst, floattype value)
    :dst(dst), value(value){

    dummy = ScalarVector(dst->size(), value);
}

Copy::Copy(Vector* dst, Vector* src)
    :dst(dst), src(src){}

DotIncMV::DotIncMV(Matrix* A, Vector* X, Vector* Y)
    :A(A), X(X), Y(Y){}

DotIncVV::DotIncVV(Vector* A, Vector* X, Vector* Y)
    :A(A), X(X), Y(Y), scalar(A->size() == 1){}

ProdUpdate::ProdUpdate(Vector* B, Vector* Y)
    :B(B), Y(Y), size(Y->size()), scalar(B->size()==1){}

Filter::Filter(Vector* input, Vector* output,
               Vector* numer, Vector* denom)
:input(input), output(output), numer(numer), denom(denom){

    for(int i = 0; i < input->size(); i++){
        x.push_back(boost::circular_buffer<floattype>(numer->size()));
        y.push_back(boost::circular_buffer<floattype>(denom->size()));
    }
}

SimLIF::SimLIF(int n_neurons, floattype tau_rc, floattype tau_ref,
               floattype dt, Vector* J, Vector* output)
:n_neurons(n_neurons), dt(dt), tau_rc(tau_rc), tau_ref(tau_ref),
dt_inv(1.0 / dt), J(J), output(output){

    voltage = ScalarVector(n_neurons, 0.0);
    refractory_time = ScalarVector(n_neurons, 0.0);
    one = ScalarVector(n_neurons, 1.0);
    dt_vec = dt * one;
}

SimLIFRate::SimLIFRate(int n_neurons, floattype tau_rc, floattype tau_ref,
                       floattype dt, Vector* J, Vector* output)
:n_neurons(n_neurons), dt(dt), tau_rc(tau_rc),
tau_ref(tau_ref), J(J), output(output){

    j = Vector(n_neurons);
    one = ScalarVector(n_neurons, 1.0);
}

// Function operator overloads

void Reset::operator() (){

    (*dst) = dummy;

    run_dbg(*this);
}

void Copy::operator() (){

    *dst = *src;

    run_dbg(*this);
}

void DotIncMV::operator() (){
    axpy_prod(*A, *X, *Y, false);

    run_dbg(*this);
}

void DotIncVV::operator() (){
    if(scalar){
        *Y += (*A)[0] * (*X);
    }else{
        (*Y)[0] += inner_prod(*A, *X);
    }

    run_dbg(*this);
}

void ProdUpdate::operator() (){
    if(scalar){
        (*Y) *= (*B)[0];
    }else{
        for (unsigned i = 0; i < size; ++i){
            (*Y)[i] *= (*B)[i];
        }
    }

    run_dbg(*this);
}

void Filter::operator() (){
    for(int i = 0; i < input->size(); i++){
        x[i].push_front((*input)[i]);

        (*output)[i] = 0.0;

        for(int j = 0; j < x[i].size(); j++){
            (*output)[i] += (*numer)[j] * x[i][j];
        }

        for(int j = 0; j < y[i].size(); j++){
            (*output)[i] -= (*denom)[j] * y[i][j];
        }

        y[i].push_front((*output)[i]);
    }

    run_dbg(*this);
}

void SimLIF::operator() (){
    dV = (dt / tau_rc) * ((*J) - voltage);
    voltage += dV;
    for(unsigned i = 0; i < n_neurons; ++i){
        voltage[i] = voltage[i] < 0 ? 0.0 : voltage[i];
    }

    refractory_time -= dt_vec;

    mult = (one - refractory_time * dt_inv);

    for(unsigned i = 0; i < n_neurons; ++i){
        mult[i] = mult[i] > 1 ? 1.0 : mult[i];
        mult[i] = mult[i] < 0 ? 0.0 : mult[i];
    }

    floattype overshoot;
    for(unsigned i = 0; i < n_neurons; ++i){
        voltage[i] *= mult[i];
        if (voltage[i] > 1.0){
            (*output)[i] = 1.0;
            overshoot = (voltage[i] - 1.0) / dV[i];
            refractory_time[i] = tau_ref + dt * (1.0 - overshoot);
            voltage[i] = 0.0;
        }
        else
        {
            (*output)[i] = 0.0;
        }
    }

    run_dbg(*this);
}

void SimLIFRate::operator() (){

    j = *J - one;

    for(unsigned i = 0; i < n_neurons; ++i){
        if(j[i] > 0.0){
            (*output)[i] = dt / (tau_ref + tau_rc * log1p(1.0 / j[i]));
        }else{
            (*output)[i] = 0.0;
        }
    }

    run_dbg(*this);
}

//to_string
string Reset::to_string() const {
    stringstream out;

    out << "Reset:" << endl;
    out << "dst:" << endl;
    out << *dst << endl;

    return out.str();
}

string Copy::to_string() const  {
    stringstream out;

    out << "Copy:" << endl;
    out << "dst:" << endl;
    out << *dst << endl;
    out << "src:" << endl;
    out << *src << endl;

    return out.str();
}

string DotIncMV::to_string() const{
    stringstream out;

    out << "DotIncMV:" << endl;
    out << "A:" << endl;
    out << *A << endl;
    out << "X:" << endl;
    out << *X << endl;
    out << "Y:" << endl;
    out << *Y << endl;

    return out.str();
}

string DotIncVV::to_string() const{
    stringstream out;

    out << "DotIncVV:" << endl;
    out << "A:" << endl;
    out << *A << endl;
    out << "X:" << endl;
    out << *X << endl;
    out << "Y:" << endl;
    out << *Y << endl;
    out << "Scalar: " << scalar << endl;

    return out.str();
}

string ProdUpdate::to_string() const{
    stringstream out;

    out << "ProdUpdate:" << endl;
    out << "B:" << endl;
    out << *B << endl;
    out << "Y:" << endl;
    out << *Y << endl;

    return out.str();
}

string Filter::to_string() const{
    stringstream out;

    out << "Filter:" << endl;

    out << "input:" << endl;
    out << *input << endl;
    out << "output:" << endl;
    out << *output << endl;
    out << "numer:" << endl;
    out << *numer << endl;
    out << "denom:" << endl;
    out << *denom << endl;

    /*
    out << "x & y:" << endl;
    for(int i = 0; i < input->size(); i++){
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
    out << *J << endl;
    out << "output:" << endl;
    out << *output << endl;
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
    out << *J << endl;
    out << "output:" << endl;
    out << *output << endl;
    return out.str();
}

