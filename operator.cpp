#include <iostream>
#include <boost/numeric/ublas/operation.hpp>

#include "operator.hpp"

Reset::Reset(Vector* dst, float value)
    :dst(dst), value(value), size(dst->size()){

    dummy = ScalarVector(size, value);
}

void Reset::operator() (){

    (*dst) = dummy;

#ifdef _DEBUG
    cout << *this;
#endif
}

ostream& operator << (ostream &out, const Reset &reset){
    out << "After Reset:" << endl;
    out << "dst:" << endl;
    out << *(reset.dst) << endl << endl;
    return out;
}

Copy::Copy(Vector* dst, Vector* src)
    :dst(dst), src(src){}

void Copy::operator() (){

    *dst = *src;

#ifdef _DEBUG
    cout << *this;
#endif
}

ostream& operator << (ostream &out, const Copy &copy){
    out << "Copy:" << endl;
    out << "dst:" << endl;
    out << *(copy.dst) << endl;
    out << "src:" << endl;
    out << *(copy.src) << endl << endl;
    return out;
}

DotInc::DotInc(Matrix* A, Vector* X, Vector* Y)
    :A(A), X(X), Y(Y){}

void DotInc::operator() (){
    axpy_prod(*A, *X, *Y, false);

#ifdef _DEBUG
    cout << *this;
#endif
}

ostream& operator << (ostream &out, const DotInc &dot_inc){
    out << "DotInc:" << endl;
    out << "A:" << endl;
    out << *(dot_inc.A) << endl;
    out << "X:" << endl;
    out << *(dot_inc.X) << endl;
    out << "Y:" << endl;
    out << *(dot_inc.Y) << endl << endl;
    return out;
}

ProdUpdate::ProdUpdate(Matrix* A, Vector* X, Vector* B, Vector* Y)
    :A(A), X(X), B(B), Y(Y), size(Y->size()){}

void ProdUpdate::operator() (){
    for (unsigned i = 0; i < size; ++i){
        (*Y)[i] *= (*B)[i];
    }
    axpy_prod(*A, *X, *Y, false);
#ifdef _DEBUG
    cout << *this;
#endif
}

ostream& operator << (ostream &out, const ProdUpdate &prod_update){
    out << "ProdUpdate:" << endl;
    out << "A:" << endl;
    out << *(prod_update.A) << endl;
    out << "X:" << endl;
    out << *(prod_update.X) << endl;
    out << "B:" << endl;
    out << *(prod_update.B) << endl << endl;
    out << "Y:" << endl;
    out << *(prod_update.Y) << endl << endl;
    return out;
}

//// needs to set up temp and pyinput
//// might be able to skip temp and directly extract to output
//PyFunc(Vector* output, boost::python::object py_fn, boost::python::object py_time, Vector* input);
//void PyFunc::operator(){
//    py_input = boost::python::extract<float>(input);
//    temp = boost::python::extract<float>(py_fn(py_time, py_input));
//    vector_assignment(output, temp);
//}

SimLIF::SimLIF(int n_neurons, float tau_rc, float tau_ref, float dt, Vector* J, Vector* output)
:n_neurons(n_neurons), dt(dt), tau_rc(tau_rc), tau_ref(tau_ref), dt_inv(1.0 / dt), J(J), output(output){
    voltage = ScalarVector(n_neurons, 0);
    refractory_time = ScalarVector(n_neurons, 0);
    one = ScalarVector(n_neurons, 1.0);
    dt_vec = dt * one;
}

void SimLIF::operator() (){
    dV = (dt / tau_rc) * (*J - voltage);
    voltage += dV;
    for(unsigned i = 0; i < n_neurons; ++i){
        voltage[i] = voltage[i] < 0 ? 0 : voltage[i];
    }

    refractory_time -= dt_vec;

    mult = (one - refractory_time * dt_inv);

    for(unsigned i = 0; i < n_neurons; ++i){
        mult[i] = mult[i] > 1 ? 1 : mult[i];
        mult[i] = mult[i] < 0 ? 0 : mult[i];
    }

    float overshoot;
    for(unsigned i = 0; i < n_neurons; ++i){
        voltage[i] *= mult[i];
        if (voltage[i] > 1){
            (*output)[i] = 1;
            overshoot = (voltage[i] - 1) / dV[i];
            refractory_time[i] = tau_ref + dt * (1 - overshoot);
            voltage[i] = 0;
        }
        else
        {
            (*output)[i] = 0;
        }
    }

#ifdef _DEBUG
    cout << *this;
#endif
}

ostream& operator << (ostream &out, const SimLIF &sim_lif){
    out << "SimLIF:" << endl;
    out << "J:" << endl;
    out << *(sim_lif.J) << endl;
    out << "output:" << endl;
    out << *(sim_lif.output) << endl;
    out << "voltage:" << endl;
    out << sim_lif.voltage << endl << endl;
    out << "refractory_time:" << endl;
    out << sim_lif.refractory_time << endl << endl;
    return out;
}

SimLIFRate::SimLIFRate(int n_neurons, float tau_rc, float tau_ref, float dt, Vector* J, Vector* output)
:n_neurons(n_neurons), dt(dt), tau_rc(tau_rc), tau_ref(tau_ref), J(J), output(output){
}

void SimLIFRate::operator() (){
    for(unsigned i = 0; i < n_neurons; ++i){
        if((*J)[i] > 0){
            (*output)[i] = dt / (tau_ref + tau_rc * log(1.0 / (*J)[i]));
        }else{
            (*output)[i] = 0.0;
        }
    }

#ifdef _DEBUG
    cout << *this;
#endif
}

ostream& operator << (ostream &out, const SimLIFRate &sim_lif_rate){
    out << "SimLIFRate:" << endl;
    out << "J:" << endl;
    out << *(sim_lif_rate.J) << endl;
    out << "output:" << endl;
    out << *(sim_lif_rate.output) << endl;
    return out;
}

MPISend::MPISend(){
}

void MPISend::operator() (){

#ifdef _DEBUG
    cout << *this;
#endif
}

ostream& operator << (ostream &out, const MPISend &mpi_send){
    return out;
}

MPIReceive::MPIReceive(){
}

void MPIReceive::operator() (){

#ifdef _DEBUG
    cout << *this;
#endif
}

ostream& operator << (ostream &out, const MPIReceive &mpi_recv){
    return out;
}
