#include "operator.hpp"

// ********************************************************************************
Reset::Reset(Signal dst, dtype value)
:dst(dst), value(value){

}

void Reset::operator() (){
    dst.fill_with(value);

    run_dbg(*this);
}

string Reset::to_string() const {

    stringstream out;
    out << Operator::to_string();
    out << "dst:" << endl;
    out << signal_to_string(dst) << endl;
    out << "value: " << value << endl;

    return out.str();
}

// ********************************************************************************
Copy::Copy(Signal dst, Signal src)
:dst(dst), src(src){

}

void Copy::operator() (){
    dst.fill_with(src);

    run_dbg(*this);
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

// ********************************************************************************
SlicedCopy::SlicedCopy(
    Signal B, Signal A, bool inc,
    int start_A, int stop_A, int step_A,
    int start_B, int stop_B, int step_B,
    vector<int> seq_A, vector<int> seq_B)
:B(B), A(A), length_A(A.shape1), length_B(B.shape1), inc(inc),
start_A(start_A), stop_A(stop_A), step_A(step_A),
start_B(start_B), stop_B(stop_B), step_B(step_B),
seq_A(seq_A), seq_B(seq_B){

    if(seq_A.size() > 0 && (start_A != 0 || stop_A != 0 || step_A != 0)){
        throw runtime_error(
            "While creating SlicedCopy, seq_A was non-empty, "
            "but one of start/step/stop was non-zero.");
    }

    if(seq_B.size() > 0 && (start_B != 0 || stop_B != 0 || step_B != 0)){
        throw runtime_error(
            "While creating SlicedCopy, seq_B was non-empty, "
            "but one of start/step/stop was non-zero.");
    }

    unsigned n_assignments_A = 0;
    if(seq_A.size() > 0){
        n_assignments_A = seq_A.size();
    }else{
        if(step_A > 0 || step_A < 0){
            n_assignments_A = unsigned(ceil(max((stop_A - start_A) / float(step_A), 0.0f)));
        }else{
            throw runtime_error("While creating SlicedCopy, step_A equal to 0.");
        }
    }

    unsigned n_assignments_B = 0;
    if(seq_B.size() > 0){
        n_assignments_B = seq_B.size();
    }else{
        if(step_B > 0 || step_B < 0){
            n_assignments_B = unsigned(ceil(max((stop_B - start_B) / float(step_B), 0.0f)));
        }else{
            throw runtime_error("While creating SlicedCopy, step_B equal to 0.");
        }
    }

    if(n_assignments_A != n_assignments_B){
        stringstream ss;
        ss << "While creating SlicedCopy, got mismatching slice sizes for A and B. "
           << "Size of A slice was " << n_assignments_A << ", while sice of B slice was "
           << n_assignments_B << "." << endl;
        throw runtime_error(ss.str());
    }

    n_assignments = n_assignments_A;
}

void SlicedCopy::operator() (){
    unsigned idx_A, idx_B;
    if(seq_A.size() > 0 && seq_B.size() > 0){
        if(inc){
            for(unsigned i = 0; i < n_assignments; i++){
                idx_A = seq_A[i] % length_A;
                idx_B = seq_B[i] % length_B;

                B(idx_B) += A(idx_A);
            }
        }else{
            for(unsigned i = 0; i < n_assignments; i++){
                idx_A = seq_A[i] % length_A;
                idx_B = seq_B[i] % length_B;

                B(idx_B) = A(idx_A);
            }
        }

    }else if(seq_A.size() > 0){
        if(inc){
            for(unsigned i = 0; i < n_assignments; i++){
                idx_A = seq_A[i] % length_A;
                idx_B = (start_B + i * step_B) % length_B;

                B(idx_B) += A(idx_A);
            }
        }else{
            for(unsigned i = 0; i < n_assignments; i++){
                idx_A = seq_A[i] % length_A;
                idx_B = (start_B + i * step_B) % length_B;

                B(idx_B) = A(idx_A);
            }
        }

    }else if(seq_B.size() > 0){
        if(inc){
            for(unsigned i = 0; i < n_assignments; i++){
                idx_A = (start_A + i * step_A) % length_A;
                idx_B = seq_B[i] % length_B;

                B(idx_B) += A(idx_A);
            }
        }else{
            for(unsigned i = 0; i < n_assignments; i++){
                idx_A = (start_A + i * step_A) % length_A;
                idx_B = seq_B[i] % length_B;

                B(idx_B) = A(idx_A);
            }
        }

    }else{
        if(inc){
            for(unsigned i = 0; i < n_assignments; i++){
                idx_A = (start_A + i * step_A) % length_A;
                idx_B = (start_B + i * step_B) % length_B;

                B(idx_B) += A(idx_A);
            }
        }else{
            for(unsigned i = 0; i < n_assignments; i++){
                idx_A = (start_A + i * step_A) % length_A;
                idx_B = (start_B + i * step_B) % length_B;

                B(idx_B) = A(idx_A);
            }
        }

    }

    run_dbg(*this);
}

string SlicedCopy::to_string() const{

    stringstream out;
    out << Operator::to_string();
    out << "B:" << endl;
    out << signal_to_string(B) << endl;
    out << "A:" << endl;
    out << signal_to_string(A) << endl;

    out << "inc: " << inc << endl;

    out << "start_A: " << start_A << endl;
    out << "stop_A:" << stop_A << endl;
    out << "step_A:" << step_A << endl;

    out << "start_B:" << start_B << endl;
    out << "stop_B:" << stop_B << endl;
    out << "step_B:" << step_B << endl;

    out << "seq_A: " << endl;
    for(int i: seq_A){
        out << i << ", ";
    }
    out << endl;

    out << "seq_B: " << endl;
    for(int i: seq_B){
        out << i << ", ";
    }
    out << endl;

    return out.str();
}

// ********************************************************************************
DotInc::DotInc(Signal A, Signal X, Signal Y)
:scalar(A.shape2 != X.shape1), matrix_vector(X.shape2 == 1), A(A), X(X), Y(Y){

    if(scalar){
        // Scalar multiplication
        bool bad_shapes =
            A.shape1 != 1 || A.shape2 != 1 || X.shape1 != Y.shape1 || X.shape2 != Y.shape2;

        if(bad_shapes){
            stringstream ss;
            ss << "While creating DotInc, got mismatching shapes for A, X and Y. "
               << "Shapes are: A - " << shape_string(A)
               << ", X - " << shape_string(X)
               << ", Y - " << shape_string(Y) << "." << endl;

            throw runtime_error(ss.str());
        }

    }else{
        // MM or MV multiplication
        bool bad_shapes =
            A.shape1 != Y.shape1 || X.shape2 != Y.shape2 || A.shape2 != X.shape1;

        if(bad_shapes){
            stringstream ss;
            ss << "While creating DotInc, got mismatching shapes for A, X and Y. "
               << "Shapes are: A - " << shape_string(A)
               << ", X - " << shape_string(X)
               << ", Y - " << shape_string(Y) << "." << endl;

            throw runtime_error(ss.str());
        }

        if(matrix_vector){
            m = A.row_major ? A.shape1 : A.shape2;
            n = A.row_major ? A.shape2 : A.shape1;
        }else{
            m = Y.shape1;
            n = Y.shape2;
            k = A.shape2;
        }

        // TODO: the requirement that A and X be contiguous can be slightly weakened.
        // All we really need is that it is stored contiguously along the major dimension.
        if(!A.is_contiguous){
            stringstream ss;
            ss << "While creating DotInc, got signal A that is not contiguous. "
               << "A: " << A << endl;

            throw runtime_error(ss.str());
        }
        transpose_A = A.row_major ? CblasNoTrans : CblasTrans;
        leading_dim_A = A.row_major ? A.stride1 : A.stride2;

        if(!X.is_contiguous){
            stringstream ss;
            ss << "While creating DotInc, got signal X that is not contiguous. "
               << "X: " << X << endl;

            throw runtime_error(ss.str());
        }
        transpose_X = X.row_major ? CblasNoTrans : CblasTrans;
        leading_dim_X = X.row_major ? X.stride1 : X.stride2;

        if(!Y.row_major){
            stringstream ss;
            ss << "While creating DotInc, got signal Y that is not in row-major order. "
               << "Y: " << Y << endl;
            throw runtime_error(ss.str());
        }
        leading_dim_Y = Y.stride1;
    }
}

void DotInc::operator() (){
    if(scalar){
        dtype a = A(0);

        for(unsigned i = 0; i < X.shape1; i++){
            for(unsigned j = 0; j < X.shape2; j++){
                Y(i, j) += a * X(i, j);
            }
        }

    }else if(X.shape2 == 1){
        cblas_dgemv(
            CblasRowMajor, transpose_A, m, n, 1.0,
            A.raw_data, leading_dim_A, X.raw_data, X.stride1,
            1.0, Y.raw_data, Y.stride1);
    }else{
        cblas_dgemm(
            CblasRowMajor, transpose_A, transpose_X, m, n, k,
            1.0, A.raw_data, leading_dim_A, X.raw_data, leading_dim_X,
            1.0, Y.raw_data, leading_dim_Y);
    }

    run_dbg(*this);
}

string DotInc::to_string() const{

    stringstream out;
    out << Operator::to_string();
    out << "scalar: " << scalar << endl;

    out << "A:" << endl;
    out << signal_to_string(A) << endl;
    out << "X:" << endl;
    out << signal_to_string(X) << endl;
    out << "Y:" << endl;
    out << signal_to_string(Y) << endl;

    return out.str();
}

// ********************************************************************************
ElementwiseInc::ElementwiseInc(Signal A, Signal X, Signal Y)
:A(A), X(X), Y(Y),
A_row_stride(A.shape1 > 1 ? 1 : 0), A_col_stride(A.shape2 > 1 ? 1 : 0),
X_row_stride(X.shape1 > 1 ? 1 : 0), X_col_stride(X.shape2 > 1 ? 1 : 0){

    if(A.shape1 != Y.shape1 && A.shape1 != 1){
        throw runtime_error(
            "While creating ElementwiseInc, A and Y had incompatible dimensions.");
    }

    if(A.shape2 != Y.shape2 && A.shape2 != 1){
        throw runtime_error(
            "While creating ElementwiseInc, A and Y had incompatible dimensions.");
    }

    if(X.shape1 != Y.shape1 && X.shape1 != 1){
        throw runtime_error(
            "While creating ElementwiseInc, X and Y had incompatible dimensions.");
    }

    if(X.shape2 != Y.shape2 && X.shape2 != 1){
        throw runtime_error(
            "While creating ElementwiseInc, X and Y had incompatible dimensions.");
    }
}

void ElementwiseInc::operator() (){
    unsigned A_i = 0, A_j = 0, X_i = 0, X_j = 0;

    for(unsigned Y_i = 0; Y_i < Y.shape1; Y_i++){
        A_j = 0;
        X_j = 0;

        for(unsigned Y_j = 0; Y_j < Y.shape2; Y_j++){
            Y(Y_i, Y_j) += A(A_i, A_j) * X(X_i, X_j);

            A_j += A_col_stride;
            X_j += X_col_stride;
        }

        A_i += A_row_stride;
        X_i += X_row_stride;
    }

    run_dbg(*this);
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

    out << "A_row_stride: " << A_row_stride << endl;
    out << "A_col_stride: " << A_col_stride << endl;

    out << "X_row_stride: " << X_row_stride << endl;
    out << "X_col_stride: " << X_col_stride << endl;

    return out.str();
}

// ********************************************************************************
NoDenSynapse::NoDenSynapse(
    Signal input, Signal output, dtype b)
:input(input), output(output), b(b){

    if(input.shape1 != output.shape1 || input.shape2 != output.shape2){
        throw runtime_error(
            "While creating NoDenSynapse, input and output had incompatible shapes.");
    }
}

void NoDenSynapse::operator() (){
    for(unsigned i = 0; i < output.shape1; i++){
        for(unsigned j = 0; j < output.shape2; j++){
            output(i, j) = b * input(i, j);
        }
    }

    run_dbg(*this);
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

// ********************************************************************************
SimpleSynapse::SimpleSynapse(Signal input, Signal output, dtype a, dtype b)
:input(input), output(output), a(a), b(b){
    if(input.shape1 != output.shape1 || input.shape2 != output.shape2){
        throw runtime_error(
            "While creating SimpleSynapse, input and output had incompatible dimensions.");
    }
}

void SimpleSynapse::operator() (){
    for(unsigned i = 0; i < output.shape1; i++){
        for(unsigned j = 0; j < output.shape2; j++){
            output(i, j) *= -a;
            output(i, j) += b * input(i, j);
        }
    }

    run_dbg(*this);
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

// ********************************************************************************
Synapse::Synapse(
    Signal input, Signal output, Signal numer, Signal denom)
:input(input), output(output), numer(numer), denom(denom){
    if(input.shape1 != output.shape1 || input.shape2 != output.shape2){
        throw runtime_error(
            "While creating Synapse, input and output had incompatible dimensions.");
    }

    for(unsigned i = 0; i < output.shape1; i++){
        for(unsigned j = 0; j < output.shape2; j++){
            x.push_back(boost::circular_buffer<dtype>(numer.shape1));
            y.push_back(boost::circular_buffer<dtype>(denom.shape1));
        }
    }
}

void Synapse::operator() (){
    unsigned idx = 0;
    for(unsigned i = 0; i < output.shape1; i++){
        for(unsigned j = 0; j < output.shape2; j++){
            x[idx].push_front(input(i, j));

            output(i, j) = 0.0;

            for(unsigned k = 0; k < x[idx].size(); k++){
                output(i, j) += numer(k) * x[idx][k];
            }

            for(unsigned k = 0; k < y[idx].size(); k++){
                output(i, j) -= denom(k) * y[idx][k];
            }

            y[idx].push_front(output(i, j));

            idx++;
        }
    }

    run_dbg(*this);
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
    for(unsigned i = 0; i < input.size(); i++){
        out << "i: " << i << endl;

        out << "x.size " << x[i].size() << endl;
        for(unsigned j = 0; j < x[i].size(); j++){
            out << "x[ "<< j << "] = "<< x[i][j] << ", ";
        }
        out << endl;

        out << "y.size " << y[i].size() << endl;
        for(unsigned j = 0; j < y[i].size(); j++){
            out << "y[ "<< j << "] = "<< y[i][j] << ", ";
        }
        out << endl;
    }
    */

    return out.str();
}

void Synapse::reset(unsigned seed){
    unsigned idx = 0;
    for(unsigned i = 0; i < output.shape1; i++){
        for(unsigned j = 0; j < output.shape2; j++){
            for(unsigned k = 0; k < x[idx].size(); k++){
                x[idx][k] = 0.0;
            }

            for(unsigned k = 0; k < y[idx].size(); k++){
                y[idx][k] = 0.0;
            }
            idx++;
        }
    }
}

// ********************************************************************************
TriangleSynapse::TriangleSynapse(
    Signal input, Signal output, dtype n0, dtype ndiff, unsigned n_taps)
:input(input), output(output), n0(n0), ndiff(ndiff), n_taps(n_taps){

    if(input.shape1 != output.shape1 || input.shape2 != output.shape2){
        throw runtime_error(
            "While creating TriangleSynapse, input and output had incompatible dimensions.");
    }

    for(unsigned i = 0; i < output.shape1; i++){
        for(unsigned j = 0; j < output.shape2; j++){
            x.push_back(boost::circular_buffer<dtype>(n_taps));
        }
    }
}

void TriangleSynapse::operator() (){
    unsigned idx = 0;
    for(unsigned i = 0; i < output.shape1; i++){
        for(unsigned j = 0; j < output.shape2; j++){
            output(i, j) += n0 * input(i, j);

            for(unsigned k = 0; k < x[idx].size(); k++){
                output(i, j) -= x[idx][k];
            }

            x[idx].push_front(ndiff * input(i, j));
            idx++;
        }
    }

    run_dbg(*this);
}

string TriangleSynapse::to_string() const{

    stringstream out;
    out << Operator::to_string();
    out << "input:" << endl;
    out << signal_to_string(input) << endl;
    out << "output:" << endl;
    out << signal_to_string(output) << endl;
    out << "n0:" << n0 << endl;
    out << "ndiff:" << ndiff << endl;
    out << "n_taps: " << n_taps << endl;

    /*
    out << "x :" << endl;
    for(unsigned i = 0; i < input.size(); i++){
        out << "i: " << i << endl;

        out << "x.size " << x[i].size() << endl;
        for(unsigned j = 0; j < x[i].size(); j++){
            out << "x[ "<< j << "] = "<< x[i][j] << ", ";
        }
        out << endl;
    }
    */

    return out.str();
}

void TriangleSynapse::reset(unsigned seed){
    unsigned idx = 0;
    for(unsigned i = 0; i < output.shape1; i++){
        for(unsigned j = 0; j < output.shape2; j++){
            for(unsigned k = 0; k < x[idx].size(); k++){
                x[idx][k] = 0.0;
            }

            idx++;
        }
    }
}

// ********************************************************************************
WhiteNoise::WhiteNoise(
    Signal output, dtype mean, dtype std, bool do_scale, bool inc, dtype dt)
:output(output), mean(mean), std(std), dist(mean, std),
alpha(do_scale ? 1.0 / dt : 1.0), do_scale(do_scale), inc(inc), dt(dt){

}

void WhiteNoise::operator() (){
    if(inc){
        for(unsigned i = 0; i < output.shape1; i++){
            output(i) += alpha * dist(rng);
        }
    }else{
        for(unsigned i = 0; i < output.shape1; i++){
            output(i) = alpha * dist(rng);
        }
    }

    run_dbg(*this);
}

string WhiteNoise::to_string() const{

    stringstream out;
    out << Operator::to_string();
    out << "output:" << endl;
    out << signal_to_string(output) << endl;
    out << "mean: " << mean << endl;
    out << "std: " << std << endl;
    out << "do_scale: " << do_scale << endl;
    out << "inc: " << inc << endl;
    out << "dt: " << dt << endl;

    return out.str();
}

void WhiteNoise::reset(unsigned seed){
    rng.seed(seed);
}

// ********************************************************************************
WhiteSignal::WhiteSignal(Signal output, Signal coefs)
:output(output), coefs(coefs), idx(0){

}

void WhiteSignal::operator() (){
    for(unsigned i = 0; i < output.shape1; i++){
        output(i) = coefs(idx % coefs.shape1, i);
    }

    idx++;

    run_dbg(*this);
}

string WhiteSignal::to_string() const{

    stringstream out;
    out << Operator::to_string();
    out << "output:" << endl;
    out << signal_to_string(output) << endl;
    out << "coefs:" << endl;
    out << signal_to_string(coefs) << endl;
    out << "idx: " << idx << endl;

    return out.str();
}

void WhiteSignal::reset(unsigned seed){
    idx = 0;
}

// ********************************************************************************
LIF::LIF(
    unsigned n_neurons, dtype tau_rc, dtype tau_ref, dtype min_voltage,
    dtype dt, Signal J, Signal output, Signal voltage,
    Signal ref_time)
:n_neurons(n_neurons), dt(dt), dt_inv(1.0 / dt), tau_rc(tau_rc), tau_ref(tau_ref),
min_voltage(min_voltage), J(J), output(output), voltage(voltage), ref_time(ref_time),
one(n_neurons, (dtype) 1.0), mult(n_neurons), dV(n_neurons){

}

void LIF::operator() (){
    // dV = -expm1(-dt / tau_rc) * (J - voltage)
    cblas_dcopy(n_neurons, J.raw_data, J.stride1, dV.raw_data, dV.stride1);
    cblas_daxpy(n_neurons, -1.0, voltage.raw_data, voltage.stride1, dV.raw_data, dV.stride1);
    dtype scale = -expm1(-dt / tau_rc);
    cblas_dscal(n_neurons, scale, dV.raw_data, dV.stride1);

    // voltage += dV
    cblas_daxpy(n_neurons, 1.0, dV.raw_data, dV.stride1, voltage.raw_data, voltage.stride1);

    // voltage = max(voltage, 0)
    dtype v;
    for(unsigned i = 0; i < n_neurons; ++i){
        v = voltage(i);
        voltage(i) = v < min_voltage ? min_voltage : v;
    }

    // ref_time -= dt_vec
    cblas_daxpy(n_neurons, -dt, one.raw_data, one.stride1,
                ref_time.raw_data, ref_time.stride1);

    // mult = -dt_inv * ref_time + 1
    cblas_dcopy(n_neurons, ref_time.raw_data, ref_time.stride1, mult.raw_data, mult.stride1);
    cblas_dscal(n_neurons, -dt_inv, mult.raw_data, mult.stride1);
    cblas_daxpy(n_neurons, 1.0, one.raw_data, one.stride1, mult.raw_data, mult.stride1);

    // mult = mult.clip(0, 1)
    dtype m;
    for(unsigned i = 0; i < n_neurons; ++i){
        m = mult(i);
        mult(i) = m > 1.0 ? 1.0 : (m < 0.0 ? 0.0 : m);
    }

    dtype overshoot;
    for(unsigned i = 0; i < n_neurons; ++i){
        voltage(i) *= mult(i);
        v = voltage(i);
        if(v > 1.0){
            output(i) = dt_inv;
            overshoot = (v - 1.0) / dV(i);
            ref_time(i) = tau_ref + dt * (1.0 - overshoot);
            voltage(i) = 0.0;
        }
        else
        {
            output(i) = 0.0;
        }
    }

    run_dbg(*this);

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

// ********************************************************************************
LIFRate::LIFRate(
    unsigned n_neurons, dtype tau_rc, dtype tau_ref, Signal J, Signal output)
:n_neurons(n_neurons), tau_rc(tau_rc), tau_ref(tau_ref), J(J), output(output){

}

void LIFRate::operator() (){
    for(unsigned i = 0; i < n_neurons; ++i){
        dtype j = J(i);
        if(j > 1.0){
             output(i) = 1.0 / (tau_ref + tau_rc * log1p(1.0 / (j - 1.0)));
        }else{
            output(i) = 0.0;
        }
    }

    run_dbg(*this);
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

// ********************************************************************************
AdaptiveLIF::AdaptiveLIF(
    unsigned n_neurons, dtype tau_n, dtype inc_n, dtype tau_rc, dtype tau_ref,
    dtype min_voltage, dtype dt, Signal J, Signal output, Signal voltage,
    Signal ref_time, Signal adaptation)
:LIF(n_neurons, tau_rc, tau_ref, min_voltage, dt, J, output, voltage, ref_time),
tau_n(tau_n), inc_n(inc_n), adaptation(adaptation), temp_J(n_neurons), dAdapt(n_neurons){

}

void AdaptiveLIF::operator() (){
    // temp_J = J
    cblas_dcopy(n_neurons, J.raw_data, J.stride1, temp_J.raw_data, temp_J.stride1);

    // J -= adaptation
    cblas_daxpy(n_neurons, -1.0, adaptation.raw_data, adaptation.stride1,
                J.raw_data, J.stride1);

    LIF::operator()();

    // J = temp_J
    cblas_dcopy(n_neurons, temp_J.raw_data, temp_J.stride1, J.raw_data, J.stride1);

    // adaptation += (dt / tau_n) * (inc_n * output - adaptation);
    cblas_dcopy(n_neurons, output.raw_data, output.stride1, dAdapt.raw_data, dAdapt.stride1);
    cblas_dscal(n_neurons, inc_n, dAdapt.raw_data, dAdapt.stride1);
    cblas_daxpy(n_neurons, -1.0, adaptation.raw_data, adaptation.stride1,
                dAdapt.raw_data, dAdapt.stride1);
    cblas_daxpy(n_neurons, dt/tau_n, dAdapt.raw_data, dAdapt.stride1,
                adaptation.raw_data, adaptation.stride1);

    run_dbg(*this);
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

// ********************************************************************************
AdaptiveLIFRate::AdaptiveLIFRate(
    unsigned n_neurons, dtype tau_n, dtype inc_n, dtype tau_rc, dtype tau_ref, dtype dt,
    Signal J, Signal output, Signal adaptation)
:LIFRate(n_neurons, tau_rc, tau_ref, J, output),
tau_n(tau_n), inc_n(inc_n), dt(dt), adaptation(adaptation),
temp_J(n_neurons), dAdapt(n_neurons){

}

void AdaptiveLIFRate::operator() (){
    // temp_J = J
    cblas_dcopy(n_neurons, J.raw_data, J.stride1, temp_J.raw_data, temp_J.stride1);

    // J -= adaptation
    cblas_daxpy(n_neurons, -1.0, adaptation.raw_data, adaptation.stride1,
                J.raw_data, J.stride1);

    LIFRate::operator()();

    // J = temp_J
    cblas_dcopy(n_neurons, temp_J.raw_data, temp_J.stride1, J.raw_data, J.stride1);

    // adaptation += (dt / tau_n) * (inc_n * output - adaptation);
    cblas_dcopy(n_neurons, output.raw_data, output.stride1, dAdapt.raw_data, dAdapt.stride1);
    cblas_dscal(n_neurons, inc_n, dAdapt.raw_data, dAdapt.stride1);
    cblas_daxpy(n_neurons, -1.0, adaptation.raw_data, adaptation.stride1,
                dAdapt.raw_data, dAdapt.stride1);
    cblas_daxpy(n_neurons, dt/tau_n, dAdapt.raw_data, dAdapt.stride1,
                adaptation.raw_data, adaptation.stride1);

    run_dbg(*this);
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

// ********************************************************************************
RectifiedLinear::RectifiedLinear(unsigned n_neurons, Signal J, Signal output)
:n_neurons(n_neurons), J(J), output(output){

}

void RectifiedLinear::operator() (){
    dtype j = 0;
    for(unsigned i = 0; i < n_neurons; ++i){
        j = J(i);
        output(i) = j > 0.0 ? j : 0.0;
    }

    run_dbg(*this);
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

// ********************************************************************************
Sigmoid::Sigmoid(unsigned n_neurons, dtype tau_ref, Signal J, Signal output)
:n_neurons(n_neurons), tau_ref(tau_ref), tau_ref_inv(1.0 / tau_ref), J(J), output(output){

}

void Sigmoid::operator() (){
    for(unsigned i = 0; i < n_neurons; ++i){
        output(i) = tau_ref_inv / (1.0 + exp(-J(i)));
    }

    run_dbg(*this);
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

// ********************************************************************************
BCM::BCM(
    Signal pre_filtered, Signal post_filtered, Signal theta,
    Signal delta, dtype learning_rate, dtype dt)
:alpha(learning_rate * dt), pre_filtered(pre_filtered), post_filtered(post_filtered),
theta(theta), delta(delta), squared_pf(post_filtered.size){

}

void BCM::operator() (){
    for(unsigned i = 0; i < post_filtered.shape1; i++){
        squared_pf(i) = post_filtered(i) * (post_filtered(i) - theta(i));
    }

    delta.fill_with(0.0);

    cblas_dger(
        CblasRowMajor, delta.shape1, delta.shape2, alpha, squared_pf.raw_data, squared_pf.stride1,
        pre_filtered.raw_data, pre_filtered.stride1, delta.raw_data, delta.stride1);

    run_dbg(*this);
}

string BCM::to_string() const{
    stringstream out;
    out << Operator::to_string();
    out << "alpha: " << alpha << endl;

    out << "pre_filtered:" << endl;
    out << signal_to_string(pre_filtered) << endl;
    out << "post_filtered:" << endl;
    out << signal_to_string(post_filtered) << endl;
    out << "theta:" << endl;
    out << signal_to_string(theta) << endl;
    out << "delta:" << endl;
    out << signal_to_string(delta) << endl;

    return out.str();
}

// ********************************************************************************
Oja::Oja(
    Signal pre_filtered, Signal post_filtered, Signal weights,
    Signal delta, dtype learning_rate, dtype dt, dtype beta)
:alpha(learning_rate * dt), beta(beta), pre_filtered(pre_filtered),
post_filtered(post_filtered), weights(weights), delta(delta){

}

void Oja::operator() (){
    for(unsigned i = 0; i < weights.shape1; i++){
        dtype post_squared = post_filtered(i);
        post_squared *= alpha * post_squared;

        for(unsigned j = 0; j < weights.shape2; j++){
            delta(i, j) = -beta * weights(i, j) * post_squared;
        }
    }

    cblas_dger(
        CblasRowMajor, delta.shape1, delta.shape2, alpha, post_filtered.raw_data, post_filtered.stride1,
        pre_filtered.raw_data, pre_filtered.stride1, delta.raw_data, delta.stride1);

    run_dbg(*this);
}

string Oja::to_string() const{
    stringstream out;
    out << Operator::to_string();
    out << "alpha: " << alpha << endl;
    out << "beta: " << beta << endl;

    out << "pre_filtered:" << endl;
    out << signal_to_string(pre_filtered) << endl;
    out << "post_filtered:" << endl;
    out << signal_to_string(post_filtered) << endl;
    out << "weights:" << endl;
    out << signal_to_string(weights) << endl;
    out << "delta:" << endl;
    out << signal_to_string(delta) << endl;

    return out.str();
}

// ********************************************************************************
Voja::Voja(
    Signal pre_decoded, Signal post_filtered, Signal scaled_encoders,
    Signal delta, Signal learning_signal, Signal scale,
    dtype learning_rate, dtype dt)
:alpha(learning_rate * dt), pre_decoded(pre_decoded), post_filtered(post_filtered),
scaled_encoders(scaled_encoders), delta(delta), learning_signal(learning_signal), scale(scale){

}

void Voja::operator() (){
    // For now, learning_signal is required to have size 1.
    dtype coef = alpha * learning_signal(0);

    for(unsigned i = 0; i < scaled_encoders.shape1; i++){
        dtype s = scale(i);
        dtype pf = post_filtered(i);

        for(unsigned j = 0; j < scaled_encoders.shape2; j++){
            delta(i, j) =
                coef * (s * post_filtered(i) * pre_decoded(j)
                - pf * scaled_encoders(i, j));
        }
    }

    run_dbg(*this);
}

string Voja::to_string() const{
    stringstream out;
    out << Operator::to_string();
    out << "alpha: " << alpha << endl;

    out << "pre_decoded:" << endl;
    out << signal_to_string(pre_decoded) << endl;
    out << "post_filtered:" << endl;
    out << signal_to_string(post_filtered) << endl;
    out << "scaled_encoders:" << endl;
    out << signal_to_string(scaled_encoders) << endl;
    out << "delta:" << endl;
    out << signal_to_string(delta) << endl;
    out << "learning_signal:" << endl;
    out << signal_to_string(learning_signal) << endl;
    out << "scale:" << endl;
    out << signal_to_string(scale) << endl;

    return out.str();
}
