from ansatz_functions import fe_1d_1o_basis_function, \
                             fe_1d_1o_basis_function_derivative, \
                             finite_element_1d_1order_basis, \
                             finite_element_1d_1order_basis_derivative
from plotting import Plot

import numpy as np
import matplotlib.pyplot as plt


omega_min = -3
omega_max = 3
x = np.linspace(omega_min, omega_max, 100)
num_elements = 10
j = 1

#phi = fe_1d_1o_basis_function(x, omega_min, omega_max, num_elements, j)
#phi = fe_1d_1o_basis_function_derivative(x, omega_min, omega_max, num_elements, j)
#plt.plot(x, phi)
#plt.show()

nodal_basis = finite_element_1d_1order_basis(x, omega_min, omega_max, num_elements)
m = nodal_basis.shape[1]
for j in range(m):
    plt.plot(x, nodal_basis[:, j])
plt.title(r'$\lambda_{j}(x)$')
plt.xlabel('x', fontsize=16)
plt.show()

nodal_basis_derivative = finite_element_1d_1order_basis_derivative(x, omega_min, omega_max, num_elements)
m = nodal_basis.shape[1]
for j in range(m):
    plt.plot(x, nodal_basis_derivative[:, j])
plt.title(r'$\lambda_{j}(x)$')
plt.xlabel('x', fontsize=16)
plt.show()
