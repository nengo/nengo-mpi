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

SimLIF::SimLIF(int n_neurons, floattype tau_rc, floattype tau_ref, floattype dt, Vector* J, Vector* output)
:n_neurons(n_neurons), dt(dt), tau_rc(tau_rc), tau_ref(tau_ref), dt_inv(1.0 / dt), J(J), output(output){
    voltage = ScalarVector(n_neurons, 0.0);
    refractory_time = ScalarVector(n_neurons, 0.0);
    one = ScalarVector(n_neurons, 1.0);
    dt_vec = dt * one;
}

SimLIFRate::SimLIFRate(int n_neurons, floattype tau_rc, floattype tau_ref, floattype dt, Vector* J, Vector* output)
:n_neurons(n_neurons), dt(dt), tau_rc(tau_rc), tau_ref(tau_ref), J(J), output(output){
    j = Vector(n_neurons);
    one = ScalarVector(n_neurons, 1.0);
}

// Function operator overloads

void Reset::operator() (){

    (*dst) = dummy;

#ifdef _DEBUG
    cout << *this;
#endif
}

void Copy::operator() (){

    *dst = *src;

#ifdef _DEBUG
    cout << *this;
#endif
}

void DotIncMV::operator() (){
    axpy_prod(*A, *X, *Y, false);

#ifdef _DEBUG
    cout << *this;
#endif
}

void DotIncVV::operator() (){
    if(scalar){
        *Y += (*A)[0] * (*X);
    }else{
        (*Y)[0] += inner_prod(*A, *X);
    }

#ifdef _DEBUG
    cout << *this;
#endif
}

void ProdUpdate::operator() (){
    if(scalar){
        (*Y) *= (*B)[0];
    }else{
        for (unsigned i = 0; i < size; ++i){
            (*Y)[i] *= (*B)[i];
        }
    }

#ifdef _DEBUG
    cout << *this;
#endif
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

#ifdef _DEBUG
    cout << *this;
#endif
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

#ifdef _DEBUG
    cout << *this;
#endif
}

//Printing
void Reset::print(ostream &out) const {
    out << "Reset:" << endl;
    out << "dst:" << endl;
    out << *dst << endl << endl;
}

void Copy::print(ostream &out) const  {
    out << "Copy:" << endl;
    out << "dst:" << endl;
    out << *dst << endl;
    out << "src:" << endl;
    out << *src << endl << endl;
}

void DotIncMV::print(ostream &out) const{
    out << "DotIncMV:" << endl;
    out << "A:" << endl;
    out << *A << endl;
    out << "X:" << endl;
    out << *X << endl;
    out << "Y:" << endl;
    out << *Y << endl;
    out << endl;
}

void DotIncVV::print(ostream &out) const{
    out << "DotIncVV:" << endl;
    out << "A:" << endl;
    out << *A << endl;
    out << "X:" << endl;
    out << *X << endl;
    out << "Y:" << endl;
    out << *Y << endl;
    out << "Scalar: " << scalar << endl;
    out << endl;
}

void ProdUpdate::print(ostream &out) const{
    out << "ProdUpdate:" << endl;
    out << "B:" << endl;
    out << *B << endl;
    out << "Y:" << endl;
    out << *Y << endl << endl;
}

void SimLIF::print(ostream &out) const{
    out << "SimLIF:" << endl;
    out << "J:" << endl;
    out << *J << endl;
    out << "output:" << endl;
    out << *output << endl;
    out << "voltage:" << endl;
    out << voltage << endl << endl;
    out << "refractory_time:" << endl;
    out << refractory_time << endl << endl;
}

void SimLIFRate::print(ostream &out) const{
    out << "SimLIFRate:" << endl;
    out << "J:" << endl;
    out << *J << endl;
    out << "output:" << endl;
    out << *output << endl;
}

