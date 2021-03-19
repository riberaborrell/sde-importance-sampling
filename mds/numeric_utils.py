import numpy as np

def arange_generator(m):
    '''this method provides a generator as a alternative to the np.arange method'''
    assert type(m) == int, ''
    assert m >= 0, ''

    n = 0
    while n < m:
        yield n
        n += 1

def coarse_vector(x, k=2):
    assert x.ndim == 1, ''
    n = x.shape[0]

    if np.mod(n, k) == 0:
        n_coa = np.floor_divide(n, k)
    else:
        n_coa = np.floor_divide(n, k) + 1

    x_coa = np.empty(n_coa)
    for i in np.arange(n_coa):
        if np.mod(n, k) != 0 and i == n_coa -1:
            x_coa[i] = np.mean(x[i*k:i*k+np.mod(n, k)])
        else:
            x_coa[i] = np.mean(x[i*k:i*k+k])
    return x_coa

def coarse_matrix(x, k=2, l=2):
    assert x.ndim == 2, ''
    n, m = x.shape

    if np.mod(n, k) == 0:
        n_coa = np.floor_divide(n, k)
    else:
        n_coa = np.floor_divide(n, k) + 1
    if np.mod(m, l) == 0:
        m_coa = np.floor_divide(m, l)
    else:
        m_coa = np.floor_divide(m, l) + 1

    x_coa = np.empty((n_coa, m_coa))
    for i in np.arange(n_coa):
        for j in np.arange(m_coa):
            if ((i != n_coa -1 and j != m_coa -1) or
                (i != n_coa -1 and np.mod(m, l) == 0) or
                (np.mod(n, k) == 0 and j != m_coa -1)):
                x_coa[i, j] = np.mean(x[i*k:i*k+k, j*l:j*l+l])
            elif i != n_coa -1 and np.mod(m, l) != 0:
                x_coa[i, j] = np.mean(x[i*k:i*k+k, j*l:j*l+np.mod(m, l)])
            elif np.mod(n, k) != 0 and j != m_coa -1:
                x_coa[i, j] = np.mean(x[i*k:i*k+np.mod(n, k), j*l:j*l+l])
            else:
                x_coa[i, j] = np.mean(x[i*k:i*k+np.mod(n, k), j*l:j*l+np.mod(m, l)])
    return x_coa

#    breakpoint()
#    x_m_coa = np.empty((n, m_coa))
#    for i in np.arange(n):
#        x_m_coa[i, :] = coarse_vector(x[i, :])
#    print(x_m_coa.shape, x_m_coa)
#    x_coa = np.empty((n_coa, m_coa))
#    for j in np.arange(m_coa):
#        print(x_m_coa[:, j])
#        print(coarse_vector(x_m_coa[:, j]))
#        print(x_coa.shape)
#        print(x_coa[:, j].shape)
#        x_coa[:, j] = coarse_vector(x_m_coa[:, j])
#    return x_coa
