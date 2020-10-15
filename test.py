from fitted_mccall_with_savings import draw_lifetime as draw_lifetime_with_savings
from fitted_mccall import draw_lifetime as draw_lifetime_without_savings
from fitted_mccall_with_savings import McCallModelContinuous as McCallModelContinuousWith
from fitted_mccall import McCallModelContinuous as Something


mcm_w = McCallModelContinuousWith()
mcm_wo = Something()

w_u, w_a, w_p = draw_lifetime_with_savings(mcm_w)
wo_u, wo_p = draw_lifetime_without_savings(mcm_wo)
for t in range(100):
    if w_u[t] != wo_u[t]:
        print(w_u[t])
        print(wo_u[t])
        print(w_p[t])
        print(wo_p[t])
        print(t)
