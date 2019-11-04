using QuadGK
using Plots
using DifferentialEquations

# Define strain rate
γd(s) = exp(-s^2)* cos(8*s)

# Stress via integration
λ = 1.
G(s) = exp(-λ*s)
σ_f(t) = quadgk(s -> γd(s)*G(t-s),-Inf,t)[1]

# Stress by solving the differential equation
f(u,p,t) = γd(t) - λ*u
u0=0.
tspan = (-10.,10.)
prob = ODEProblem(f,u0,tspan)
sol = solve(prob,Tsit5(),reltol=1e-8,abstol=1e-8)

# Behold they are the same!
plot(sol)
t_plot = -3.:0.02:3.
plot!(t_plot,σ_f.(t_full),xlims = (-3.,3.))
