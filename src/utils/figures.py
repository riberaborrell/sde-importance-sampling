import matplotlib.pyplot as plt

COLORS_TAB10 = [plt.cm.tab10(i) for i in range(20)]

COLORS_TAB20 = [plt.cm.tab20(i) for i in range(20)]

COLORS_TAB20b = [plt.cm.tab20b(i) for i in range(20)]

COLORS_TAB20c = [plt.cm.tab20c(i) for i in range(20)]

COLORS_FIG = {
    'mc-sampling': COLORS_TAB20[0],     # blue
    'mc-sampling-2': COLORS_TAB20[1],   # light blue
    'optimal-is': COLORS_TAB20[4],      # green 
    'optimal-is-2': COLORS_TAB20[5],    # light green
    'hjb-solution': COLORS_TAB20[14],   # grey
    'metadynamics': COLORS_TAB20[6],    # red
    'gaussian': COLORS_TAB20[10],       # brown
    'gaussian-init': COLORS_TAB20[2],   # orange
    'nn': COLORS_TAB20[8],              # lila
    'nn-init': COLORS_TAB20[12],        # pink 
    'nn-init-cv': COLORS_TAB20[10],     # brown
}

TITLES_FIG = {
    'potential': r'Potential $V_\alpha$',
    'control': r'Control $u_\theta$',
    'control-meta': r'Control $u^{meta}(x)$',
    'optimal-control': r'Optimal control $u^*$',
    'perturbed-potential': r'Perturbed potential $V + V_{bias}$',
    'perturbed-potential-meta': r'Perturbed potential $V + V_{bias}^{meta}$',
    'optimal-potential': r'Optimal potential $V + V_{bias}^*$',
    'psi': r'$\widehat{\Psi}$',
    'psi-meta': r'$\widehat{\Psi}^{meta}$',
    'value-function': r'$\widehat{\Phi}$',
    'value-function-meta': r'$\widehat{\Phi}^{meta}$',
    'var-i': r'$\widehat{Var}(I)$',
    're-i': r'$\widehat{RE}(I)$',
    'var-i-u': r'$\widehat{Var}(I^u)$',
    're-i-u': r'$\widehat{RE}(I^u)$',
    'loss': r'$\widehat{J}(u_\theta; x)$',
    'var-loss': r'$\widehat{Var}(\widetilde{J}(\theta; x_0))$',
    'u-l2-error': 'Estimation of $L^2$-error',
    'time-steps': r'TS',
    'ct': r'CT(s)',
}

LABELS_FIG = {
    'grad-steps': 'gradient iterations',
    'ct': 'CT(s)',
}
