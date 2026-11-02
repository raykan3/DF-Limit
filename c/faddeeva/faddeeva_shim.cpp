/* faddeeva_shim.cpp: C entry points to the Faddeeva package (C++), so that the engine's C sources compile with every
 * compiler, MSVC included (whose C has no `double complex` syntax for the package's C header). */
#include "Faddeeva.hh"
extern "C" void dfl_faddeeva_w(double re, double im, double *ore, double *oim)
{
    std::complex<double> r = Faddeeva::w(std::complex<double>(re, im), 0.0);
    *ore = r.real(); *oim = r.imag();
}
extern "C" void dfl_faddeeva_erf(double re, double im, double *ore, double *oim)
{
    std::complex<double> r = Faddeeva::erf(std::complex<double>(re, im), 0.0);
    *ore = r.real(); *oim = r.imag();
}
