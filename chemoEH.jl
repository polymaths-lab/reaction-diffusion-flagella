using LinearAlgebra
using DelimitedFiles
using SparseArrays
using DifferentialEquations
using Sundials
using FFTW
using Statistics


n = 100

mu = 100
lamda = 0

eta=0.34
zeta=0.6

fstar=2

eps_flagellum = 0.003
eps_major = 0.077
eps_minor = 0.048

Th0, ThN, nP0, nPN, nM0, nMN, N0, NN, T0, TN  = 3, n+3, n+4, 2*n+4, 2*n+5, 3*n+5, 3*n+6, 4*n+6, 4*n+7, 5*n+7

# Finite difference derivative functions

F(X) = n*[-1.5X[1] + 2X[2] - 0.5X[3];
          collect(0.5*(X[i+1] - X[i-1] for i in 2:n));
          0.5X[n-1] - 2X[n] + 1.5X[n+1]]

S(X) = n^2*[2*X[1] -5*X[2] + 4*X[3] - X[4];
            collect(X[i-1] - 2*X[i] + X[i+1] for i in 2:n);
            -X[n-2] + 4*X[n-1] - 5*X[n] + 2*X[n+1]]

FORWARD(x) = n*(-1.5*x[1] + 2*x[2] - 0.5*x[3])
BACKWARD(x) = n*(0.5*x[n-1] - 2*x[n] + 1.5*x[n+1])
INTEGRAL(x) = (0.5*x[1] + sum(x[i] for i in 2:n) + 0.5*x[n+1])/n

cT(e) = 8*e^3/(3*(-2*e + (1 + e^2)*log((1+e)/(1-e))))
cN(e) = 16*e^3/(3*(2*e + (3*e^2 - 1)*log((1+e)/(1-e))))
cOmega(e) = 4*e^3*(2 - e^2)/(3*(1 - e^2)*(-2*e + (e^2 + 1)*log((1+e)/(1-e))))
eccentricity(a, b) = sqrt(a^2 - b^2)/a

e = eccentricity(eps_major, eps_minor)

# Initial conditions
s = 0:1/n:1
gaussian(sigma) = 0.001.*exp.(-(s.-0.5).^2 / sigma^2)

function bvp(theta, nplus, nminus, Sp4, mu_a, Sp_N4, Sp_T4, Sp_Omega4)
    D1 = sparse(0.5*n*Tridiagonal(-ones(n), zeros(n+1), ones(n)))
    D2 = sparse(n^2*Tridiagonal(ones(n), -2ones(n+1), ones(n)))
    D1_0 = n*[-1.5 2 -0.5 zeros(n-2)']
    D1_1 = n*[zeros(n-2)' 0.5 -2  1.5]
    Delta = theta .- theta[1]
    theta_s = D1 * theta
    theta_ss = D2 * theta

    Z = -zeta*mu_a*(nplus + nminus)
    theta_t_0 = sparse(1:n+1, ones(n+1), Z, n+1, n+1)
    Q = spdiagm(0 => Z) - theta_t_0
    O = zeros(n+1,n+1)
    Id = Diagonal(ones(n+1))

    A = [Sp4         0           zeros(n+1)'   [2FORWARD(theta) zeros(n)']            -2*D1_0
         0           Sp4         zeros(n+1)'   -D1_0                                  [-FORWARD(theta) zeros(n)']
         zeros(n+1)  zeros(n+1)  -Sp4*Id       D2 - 2Diagonal(theta_s.^2)             3*theta_s.*D1 + Diagonal(theta_ss)
         zeros(n+1)  zeros(n+1)  Q - lamda*Id  Id                                     O
         zeros(n+1)  zeros(n+1)  O             -3*theta_s.*D1 - 2Diagonal(theta_ss)   2*D2 - Diagonal(theta_s.^2) ]

    b = [zeros(n+4); mu*Delta - mu_a*(nplus - nminus) - theta_ss; zeros(n)]

    th0, thN, n0, nN, t0, tN = 3, n+3, n+4, 2*n+4, 2*n+5, 3*n+5

    B = zero(A)
    B[th0,2] = -Sp_N4
    B[th0,th0] = Sp_N4*eps_major
    B[th0,n0] = 1
    B[thN,thN-2] = 0.5*n
    B[thN,thN-1] = -2*n
    B[thN,thN] = 1.5*n
    B[n0,th0] = -Sp_Omega4
    B[n0,n0] = eps_major + 0.5/n
    B[n0,n0+1:nN-1] .= 1/n
    B[n0,nN] = 0.5/n
    B[nN,nN] = n
    B[t0,t0] = 1
    B[t0,1] = -Sp_T4
    B[tN,tN] = n

    ZEROROWS = ones(tN)
    ZEROROWS[[th0, thN, n0, nN, t0, tN]] .= 0
    #BCS = sparse(rows, cols, data, tN, tN)
    A = (ZEROROWS .* A) + B

    # Solve the linear system with a sparse solver
    sol = A \ b

    return sol[1], sol[2], sol[th0:thN], sol[n0:nN], sol[t0:tN]
end

residuals = include("sys_fn$(n).jl")
Jac_fn = include("J$(n).jl")

differential_vars = [trues(3*n+5); falses(2*n+2)]
differential_vars[ThN] = false

pattern = sparse(include("pattern$(n).jl"))



function solve_dae(Sp, mu_a)
    Sp4 = Sp^4
    Sp_T4 = 1.5*eps_major*log(1/eps_flagellum)*cT(e)*Sp4
    Sp_N4 = 1.5*eps_major*log(1/eps_flagellum)*cN(e)*Sp4
    Sp_Omega4 = 2*eps_major*eps_minor^2*log(1/eps_flagellum)*cOmega(e)*Sp4
    
    # Initial Condition

    theta_0 = gaussian(0.1)
    nP_0 = 0.02*ones(n+1); nM_0 = 0.02*ones(n+1)
    xt_0, yt_0, thetat_0, N_0, T_0 = bvp(theta_0, nP_0, nM_0, Sp4, mu_a, Sp_N4, Sp_T4, Sp_Omega4)
    #xt_0, yt_0, thetat_0, N_0, T_0 = 0, 0, zeros(n+1), zeros(n+1), zeros(n+1)
    X_0 = [0; 0; theta_0; nP_0; nM_0; N_0; T_0]

    Delta_t_0, nPt_0, nMt_0 = zeros(n+1), zeros(n+1), zeros(n+1)
    Xt_0 = [xt_0; yt_0; thetat_0; nPt_0; nMt_0; zeros(n+1); zeros(n+1)]

    Deltat_0 = thetat_0 .- thetat_0[1]
    nPt_0 = eta*(1 .- nP_0) - (1-eta)*nP_0.*exp.(fstar*(1 .- zeta*Deltat_0))
    nMt_0 = eta*(1 .- nM_0) - (1-eta)*nM_0.*exp.(fstar*(1 .+ zeta*Deltat_0))
    Xt_0 = [xt_0; yt_0; thetat_0; nPt_0; nMt_0; zeros(n+1); zeros(n+1)]
    
    # Solve dae
    
    tugofwar!(out, X_t, X, p, t) = residuals(out, [Sp4; mu_a; X; X_t])
    jac!(J, X_t, X, p, gamma, t) = Jac_fn(J, [gamma; Sp4; mu_a; X; X_t])
    tow = DAEFunction(tugofwar!, jac=jac!, jac_prototype=pattern)
    prob = DAEProblem(tow, Xt_0, X_0, [0.0, 60.0],
                      differential_vars=differential_vars,
                      saveat=0.01)
    sol = solve(prob, IDA(linear_solver=:KLU), maxiters=1000000);
    x, y, theta, nplus, nminus = sol[1,:], sol[2,:], sol[Th0:ThN,:], sol[nP0:nPN,:], sol[nM0:nMN,:]
    N, T = sol[N0:NN,:], sol[T0:TN,:]
    #delta = theta .- theta[1,:]'
    #println(sol.t)
    #return x, y, theta, nplus, nminus, N, T
    return sol
end

########################################################################################
# Calculate quantities

function dynamic_mode(delta)
    Z = rfft(delta, 2) ./ size(delta,2)
    ps = mean(abs.(Z).^2, dims=1)'
    freqs = rfftfreq(size(delta, 2), 20)
    _, i = findmax(ps)
    delta1 = Z[:,i]
    omega = 2*pi*freqs[i]
    return delta1, omega
end

function cycle(x, y, theta)
    _, omega = dynamic_mode((theta .- theta[1,:]')[:,end-400:end]) 
    T = floor(Int, 2*pi/(0.05*omega))
    step = (10 < T) ? div(T, 10) : 1
    return x[end-T:step:end], y[end-T:step:end], theta[:,end-T:step:end]
end

function rotate_pts(x, y, theta)
    theta_mean = mean(theta)
    x_t = x .- x[1]
    y_t = y .- y[1]
    x_rot = x_t*cos(theta_mean) + y_t*sin(theta_mean)
    y_rot = -x_t*sin(theta_mean) + y_t*cos(theta_mean)
    y_rot .-= mean(y_rot)
    return x_rot, y_rot, theta .- theta_mean
end
    
VSL(x_r) = abs(x_r[end])/(400*0.05)

VSLs = []
delta_stds = []
N_stds = []
N_rats = []