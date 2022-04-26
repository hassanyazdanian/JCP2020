In this directory, there are two benchmark tests in order to validate the
technique developed to compute the Biot-Savart integral. In both tests, the 
magnetic flux density is known analytically, and therefore the performance of
the numerical algorithm can be compared to a ground truth. The first example is
a smooth phantom, while the second phantom is piecewise smooth.

$$  A_1 (\mathbf x) &= x_2x_3 \exp\left(\dfrac{-1}{h^2-| \mathbf{x}|^2}\right), \\[0.1in]  
	A_2 (\mathbf x) &=x_1x_3 \exp\left(\dfrac{-1}{h^2-| \mathbf{x}|^2}\right), \\[0.1in]
	A_3 (\mathbf x) &=-2x_1x_2 \exp\left(\dfrac{-1}{h^2-| \mathbf{x}|^2}\right)$$

To illustrate the superiority of the FFT-based method over the direct 
discretization of the Biot-Savart integral, we also numerically evaluated 
the Biot-Savart integral using Simpson’s rule.