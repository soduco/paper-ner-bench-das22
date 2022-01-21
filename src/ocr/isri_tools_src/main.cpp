#include <pybind11/pybind11.h>

void init_align(pybind11::module* m);
void init_accuracy(pybind11::module& m);

PYBIND11_MODULE(isri_tools, m) {
    m.doc() = "ISRI Analytic tools"; // optional module docstring
    init_align(&m);
    init_accuracy(m);
}
