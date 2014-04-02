#include <iostream>
#include <boost/numeric/ublas/operation.hpp>

#include "operator.hpp"

Reset::Reset(Vector* dst, float value)
    :dst(dst), value(value), size(dst->size()){

    dummy = scalar_vector<double>(size, value);

}

void Reset::operator() (){

    (*dst) = dummy;

#ifdef _DEBUG
    std::cout << "After Reset:" << std::endl;
    std::cout << "dst:" << std::endl;
    std::cout << *dst << std::endl << std::endl;
#endif
}

Copy::Copy(Vector* dst, Vector* src)
    :dst(dst), src(src){}

void Copy::operator() (){

    *dst = *src;

#ifdef _DEBUG
    std::cout << "After Copy:" << std::endl;
    std::cout << "dst:" << std::endl;
    std::cout << *dst << std::endl;
    std::cout << "src:" << std::endl;
    std::cout << *src << std::endl << std::endl;
#endif
}

DotInc::DotInc(Matrix* A, Vector* X, Vector* Y)
    :A(A), X(X), Y(Y){}

void DotInc::operator() (){
    axpy_prod(*A, *X, *Y, false);

#ifdef _DEBUG
    std::cout << "After DotInc:" << std::endl;
    std::cout << "A:" << std::endl;
    std::cout << *A << std::endl;
    std::cout << "X:" << std::endl;
    std::cout << *X << std::endl;
    std::cout << "Y:" << std::endl;
    std::cout << *Y << std::endl << std::endl;
#endif
}

ProdUpdate::ProdUpdate(Matrix* A, Vector* X, Vector* B, Vector* Y)
    :A(A), X(X), B(B), Y(Y), size(Y->size()){}

void ProdUpdate::operator() (){
    for (unsigned i = 0; i < size; ++i){
        (*Y)[i] *= (*B)[i];
    }
    axpy_prod(*A, *X, *Y, false);
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
    voltage = scalar_vector<double>(n_neurons, 0);
    refractory_time = scalar_vector<double>(n_neurons, 0);
    one = scalar_vector<double>(n_neurons, 1.0);
    dt_vec = dt * one;
}

//Reference Python Math:
//dV = (dt / self.tau_rc) * (J - voltage)
//voltage += dV
//voltage[voltage < 0] = 0  # clip values below zero
//refractory_time -= dt
//voltage *= (1 - refractory_time / dt).clip(0, 1)
//output[:] = (voltage > 1)
//overshoot = (voltage[output > 0] - 1) / dV[output > 0]
//spiketime = dt * (1 - overshoot)
//voltage[output > 0] = 0
//refractory_time[output > 0] = self.tau_ref + spiketime
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
}

SimLIFRate::SimLIFRate(int n_neurons, float tau_rc, float tau_ref, float dt, Vector* J, Vector* output)
:n_neurons(n_neurons), dt(dt), tau_rc(tau_rc), tau_ref(tau_ref), J(J), output(output){
}

//Reference Python Math:
//j = J - 1
//output[:] = 0  # faster than output[j <= 0] = 0
//output[j > 0] = dt / (
//    self.tau_ref + self.tau_rc * np.log1p(1. / j[j > 0]))
void SimLIFRate::operator() (){
    for(unsigned i = 0; i < n_neurons; ++i){
        if((*J)[i] > 0){
            (*output)[i] = dt / (tau_ref + tau_rc * log(1.0 / (*J)[i]));
        }else{
            (*output)[i] = 0.0;
        }
    }
}

MPISend::MPISend(){
}

void MPISend::operator() (){
}

MPIReceive::MPIReceive(){
}

void MPIReceive::operator() (){
}
