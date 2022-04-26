In this directory, there are two benchmark tests in order to validate the
technique developed to compute the Biot-Savart integral. In both tests, the 
magnetic flux density is known analytically, and therefore the performance of
the numerical algorithm can be compared to a ground truth. The first example is
a smooth phantom, while the second phantom is piecewise smooth.

$$\int_\Omega \nabla u \cdot \nabla v~dx = \int_\Omega fv~dx$$

To illustrate the superiority of the FFT-based method over the direct 
discretization of the Biot-Savart integral, we also numerically evaluated 
the Biot-Savart integral using Simpson’s rule.