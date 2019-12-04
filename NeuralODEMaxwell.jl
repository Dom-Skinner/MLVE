using DiffEqFlux, Flux, DifferentialEquations
using Plots
using ForwardDiff


function dudt_exact(u,p,t,γd)
    σ = u[1]
    λ = p[1]
    dσ = γd(t) -λ*σ
    return [dσ]
end
function get_exact_sol(tspan,tsave,u0,p,γd)
    prob_exact = ODEProblem((u,p,t) -> dudt_exact(u,p,t,γd),u0,tspan,p)
    sol_exact = solve(prob_exact,Tsit5(),saveat=tsave)
    return hcat(sol_exact.u...)
end
## --- Reverse-mode AD ---

tspan = (0.0f0,10.0f0)
tsave = range(tspan[1],tspan[2],length=80)

γd = t -> cos.(2.0f0.*t) + cos.(3.0f0.*t)
γdd = t -> ForwardDiff.derivative(γd, t)

σ0 =0.248228f0
λ = 2.0f0
σ_exact = get_exact_sol(tspan,tsave,[σ0],[λ],γd)[1,:]


p = []
u0 = Tracker.param([σ0; γd(0.0f0)])


f0 = Chain(Dense(2,10,tanh), Dense(10,1))
f1 = Chain(Dense(2,10,tanh), Dense(10,1))



function dudt_(u::TrackedArray,p,t)
    x, y = u
    Flux.Tracker.collect(
        [f1(u)[1],
        γdd(t)])
end
function dudt_(u::AbstractArray,p,t)
    x, y = u
    [Flux.data(f1(u)[1]),
    γdd(t)]
end

prob = ODEProblem(dudt_,u0,tspan,p)
diffeq_rd(p,prob,Tsit5())

function predict_rd()
  Flux.Tracker.collect(diffeq_rd(p,prob,Tsit5(),u0=u0,saveat=tsave))
end
#loss_rd() =   sum( (predict_rd() .- sol_exact_arr).^2 )
loss_rd() =   begin
    P_RD = predict_rd()#[1,:]
    σ_out = [f0(P_RD[:,i])[1] for i = 1:size(P_RD,2)] # this is dumb
    return sum( (σ_out .- σ_exact).^2 )
end
loss_rd()

data = Iterators.repeated((), 20)
opt = ADAM(0.015)
cb = function ()
  display(loss_rd())

  ode_solve = solve(remake(prob,u0=Flux.data(u0),p=Flux.data(p)),Tsit5(),saveat=0.1)

  f0_ = Flux.data(f0)
  ϕ = [f0_(o).data[1] for o in ode_solve.u]
  plt = plot(ode_solve.t,ϕ,m=:circle,
  #ylim=(-0.58,0.58),
  label="NN solution",ylabel="stress",xlabel="time")
  display(plot!(plt,tsave,σ_exact,label="True solution"))

end
cb = function () end

# Display the ODE with the current parameter values.
cb()
Flux.train!(loss_rd, params(f0,f1,p,u0), data, opt, cb = cb)

savefig("/Users/Dominic/Dropbox (MIT)/18.337 Final Project/NonLin_training.pdf")




tspan = (0.0f0,10.0f0)
tsave = range(tspan[1],tspan[2],length=80)

γd = t -> 1.2*cos.(1.0f0.*t) + 0.8*cos.(3.0f0.*t)
γdd = t -> ForwardDiff.derivative(γd, t)

σ0 =0.248228f0
λ = 2.0f0
σ_exact = get_exact_sol(tspan,tsave,[σ0],[λ],γd)[1,:]


function dudt_(u::TrackedArray,p,t)
    x, y = u
    Flux.Tracker.collect(
        [f1(u)[1],
        γdd(t)])
end
function dudt_(u::AbstractArray,p,t)
    x, y = u
    [Flux.data(f1(u)[1]),
    γdd(t)]
end

prob = ODEProblem(dudt_,u0,tspan,p)

  ode_solve = solve(remake(prob,u0=Flux.data(u0),p=Flux.data(p)),Tsit5(),saveat=0.1)

  f0_ = Flux.data(f0)
  ϕ = [f0_(o).data[1] for o in ode_solve.u]
  plt = plot(ode_solve.t,ϕ,m=:circle,
  #ylim=(-0.58,0.58),
  label="NN solution",ylabel="stress",xlabel="time")
  display(plot!(plt,tsave,σ_exact,label="True solution"))


savefig("/Users/Dominic/Dropbox (MIT)/18.337 Final Project/NonLin_testing.pdf")
