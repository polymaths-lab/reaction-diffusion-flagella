using Symbolics
using SparseArrays
n = 100
@variables Sp⁴ μₐ x y xₜ yₜ γ ϵ
@variables θ[1:n+1] θₜ[1:n+1] p[1:n+1] q[1:n+1] pₜ[1:n+1] qₜ[1:n+1]
@variables N[1:n+1] Nₜ[1:n+1] T[1:n+1] Tₜ[1:n+1]

μ = 100
#η = 0.14
η = 0.34
ζ = 0.6
#ζ = 0.3
f = 2
λ = 0
#mu_a = 16000
ϵ_body = 0.003
ϵ = 0.077
ϵ_minor = 0.048
# e = 0.7819217258956036
# ϵ_minor = ϵ*sqrt(1 - e^2)

cT(e) = 8*e^3/(3*(-2*e + (1 + e^2)*log((1+e)/(1-e))))
cN(e) = 16*e^3/(3*(2*e + (3*e^2 - 1)*log((1+e)/(1-e))))
cΩ(e) = 4*e^3*(2 - e^2)/(3*(1 - e^2)*(-2*e + (e^2 + 1)*log((1+e)/(1-e))))
eccentricity(a, b) = sqrt(a^2 - b^2)/a

e = eccentricity(ϵ, ϵ_minor)
Sp_T = 1.5*ϵ*log(1/ϵ_body)*cT(e)*Sp⁴
Sp_N = 1.5*ϵ*log(1/ϵ_body)*cN(e)*Sp⁴
Sp_Omega = 2*ϵ*ϵ_minor^2*log(1/ϵ_body)*cΩ(e)*Sp⁴


d_s(Q, m) = Q[m+1] - Q[m-1]
d_ss(Q, m) = Q[m+1] - 2*Q[m] + Q[m-1]

vt = xₜ*cos(θ[1]) + yₜ*sin(θ[1])
vn = -xₜ*sin(θ[1]) + yₜ*cos(θ[1])

sys = [Sp⁴*vt + n*(3*T[1] - 4*T[2] + T[3] - N[1]*(3*θ[1] - 4*θ[2] + θ[3]))
       Sp⁴*vn + n*(T[1]*(1.5*θ[1] - 2*θ[2] + 0.5*θ[3]) + 1.5*N[1] - 2*N[2] + 0.5*N[3])
       N[1] - Sp_N*(vn - ϵ*θₜ[1])
       [Sp⁴*θₜ[m] - n^2*d_ss(N,m) + 0.5*n^2*N[m]*d_s(θ,m)^2 - 0.75*n^2*d_s(T,m)*d_s(θ,m) - n^2*T[m]*d_ss(θ,m) 
        for m in 2:n]
       N[end]
       n^2*[pₜ[m] - η*(1 - p[m]) + (1-η)*p[m]*exp(f*(1 - ζ*(θₜ[m] - θₜ[1]))) for m in 1:n+1]
       n^2*[qₜ[m] - η*(1 - q[m]) + (1-η)*q[m]*exp(f*(1 + ζ*(θₜ[m] - θₜ[1]))) for m in 1:n+1]
       (0.5*N[1] + sum(N[m] for m in 2:n) + 0.5*N[end])/n - Sp_Omega*θₜ[1] + ϵ*N[1]
       [N[m] + n^2*d_ss(θ,m) - μ*(θ[m] - θ[1]) - λ*(θₜ[m] - θₜ[1]) + μₐ*(p[m] - q[m]) - μₐ*ζ*(p[m] + q[m])*(θₜ[m] - θₜ[1])
        for m in 2:n]
       n*(0.5*θ[end-2] - 2*θ[end-1] + 1.5*θ[end])
       T[1] - Sp_T*vt
       [2*n^2*d_ss(T,m) - 0.75*n^2*d_s(N,m)*d_s(θ,m) - 2*n^2*N[m]*d_ss(θ,m) - 0.25*n^2*T[m]*d_s(θ,m)^2 for m in 2:n]
       T[end]]

X = [x; y; θ; p; q; N; T]
Xₜ = [xₜ; yₜ; θₜ; pₜ; qₜ; Nₜ; Tₜ]     

vars = [Sp⁴; μₐ; X; Xₜ]
J_X = Symbolics.sparsejacobian(sys, X)
J_Xₜ = Symbolics.sparsejacobian(sys, Xₜ)
pattern_X = Symbolics.jacobian_sparsity(sys, X)
pattern_Xₜ = Symbolics.jacobian_sparsity(sys, Xₜ)
pattern = pattern_X + pattern_Xₜ
write("pattern$(n).jl", string(Matrix(pattern)))

Jac = J_X + γ*J_Xₜ
sys_expr = build_function(sys, vars)
Jac_expr = build_function(Jac, [γ; vars])

write("sys_fn$(n).jl", string(sys_expr[2]))
write("J$(n).jl", string(Jac_expr[2]))
