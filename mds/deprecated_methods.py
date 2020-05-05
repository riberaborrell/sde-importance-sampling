def sample_not_drifted_not_vectorized(self):
    M = self.M
    N = self.N
    dt = self.dt
    xzero = self.xzero
    beta = self.beta
    target_set_min = self.target_set_min
    target_set_max = self.target_set_max

    self.preallocate_variables(is_sampling_problem=True)

    for i in np.arange(M):
        # initialize Xtemp
        Xtemp = xzero
        
        for n in np.arange(1, N+1):
            # Brownian increment
            dB = np.sqrt(dt) * np.random.normal(0, 1)

            # compute gradient
            gradient = double_well_1d_gradient(Xtemp)

            # SDE iteration
            drift = - gradient * dt
            diffusion = np.sqrt(2 / beta) * dB
            Xtemp = Xtemp + drift + diffusion
            
            # check if we have arrived to the target set
            if (Xtemp >= target_set_min and Xtemp <= target_set_max):
                fht = n * dt

                # save first hitting time
                self.fht[i] = fht

                # save quantity of interest at the fht
                self.Psi[i] = np.exp(-beta * fht)
                break


def sample_drifted_not_vectorized(self):
    
    M = self.M
    N = self.N
    dt = self.dt
    xzero = self.xzero
    beta = self.beta
    target_set_min = self.target_set_min
    target_set_max = self.target_set_max
    
    self.preallocate_variables(is_sampling_problem=True)

    k = 100

    for i in np.arange(M):
        # initialize Xtemp
        Xtemp = xzero
        
        # initialize Girsanov Martingale terms, G_t = e^(G1_t + G2_t)
        G1temp = 0
        G2temp = 0
        
        for n in np.arange(1, N+1):
            # Brownian increment
            dB = np.sqrt(dt) * np.random.normal(0, 1)
            
            # compute control at Xtemp
            utemp = self.control(Xtemp)

            # compute gradient
            gradient = self.tilted_gradient(Xtemp, utemp)

            # SDE iteration
            drift = - gradient * dt
            diffusion = np.sqrt(2 / beta) * dB
            Xtemp = Xtemp + drift + diffusion
            
            # compute Girsanov Martingale terms
            # G1_t = int_0^fht -u_t dB_t
            # G2_t = int_0^fht - 1/2 (u_t)^2 dt
            G1temp = G1temp - np.sqrt(1 / beta) * utemp * dB
            G2temp = G2temp - (1 / beta) * 0.5 * (utemp ** 2) * dt 
            
            # save Girsanov Martingale at time k
            if n == k: 
                self.G_N[i] = np.exp(G1temp + G2temp)

            # check if we have arrived to the target set
            if (Xtemp >= target_set_min and Xtemp <= target_set_max):
                fht = n * dt

                # save first hitting time
                self.fht[i] = fht

                # save Girsanov Martingale at time k
                self.G_fht[i] = np.exp(G1temp + G2temp)

                # save re-weighted quantity of interest
                self.Psi_rew[i] = np.exp(-beta * fht + G1temp + G2temp) 
                break


def sample_soc_not_vectorized(self):
    M = self.M
    N = self.N
    dt = self.dt
    xzero = self.xzero
    beta = self.beta
    target_set_min = self.target_set_min
    target_set_max = self.target_set_max
    m = self.m
    
    self.preallocate_variables(is_soc_problem=True)

    for i in np.arange(M):
        # initialize Xtemp
        Xtemp = xzero
        
        # compute control at Xtemp
        utemp = self.control(Xtemp)

        # bla
        cost = 0
        sum_partial_tilde_gh = np.zeros(m)
        grad_Sh = np.zeros(m)
        
        for n in np.arange(1, N+1):
            # Brownian increment
            dB = np.sqrt(dt) * np.random.normal(0, 1)

            # compute gradient
            gradient = self.tilted_gradient(Xtemp, utemp)

            # SDE iteration
            drift = - gradient * dt
            diffusion = np.sqrt(2 / beta) * dB
            Xtemp = Xtemp + drift + diffusion
            
            # compute control at Xtemp
            utemp = self.control(Xtemp)

            # evaluate the control basis functions at Xtmep
            btemp = self.control_basis_functions(Xtemp)
                
            # compute cost, ...
            cost = cost + 0.5 * (utemp ** 2) * dt
            sum_partial_tilde_gh = sum_partial_tilde_gh + utemp * btemp * dt  
            grad_Sh = grad_Sh + np.random.normal(0, 1) * btemp
            
            # check if we have arrived to the target set
            if (Xtemp >= target_set_min and Xtemp <= target_set_max):
                fht = n * dt

                # save first hitting time
                self.fht[i] = fht

                self.J[i] = cost + fht
                grad_Sh = grad_Sh * (- np.sqrt(dt * beta / 2))
                self.gradJ[i, :] = sum_partial_tilde_gh - (cost + fht) * grad_Sh
                
                break
