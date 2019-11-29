using DiffEqFlux, Flux, DifferentialEquations,Plots

## --- Reverse-mode AD ---

tspan = (-25.0f0,25.0f0)
u0 = Tracker.param(Float32[0.8; 0.8])
t_save = range(tspan[1],tspan[2],length=30)

ann = Chain(Dense(2,10,tanh), Dense(10,1))
p = param(Float32[-2.0,1.1])


## ---- Create true ode data ----
#γd(t) = exp(-0.05*t^2)
γdd(t) = -0.1*t*exp(-0.05*t^2)
function trueODEfunc(du,u,p,t)
    du[1] = -1.1*u[1] - u[2]
    du[2] = γdd(t)
end
prob = ODEProblem(trueODEfunc,[0.8; 0.8],tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t_save))



function dudt_(u::TrackedArray,p,t)
    x, y = u
    Flux.Tracker.collect(
        [ann(u)[1],
        γdd(t)])
end
function dudt_(u::AbstractArray,p,t)
    x, y = u
    [Flux.data(ann(u)[1]),
    γdd(t)]
end

prob = ODEProblem(dudt_,u0,tspan,p)
diffeq_rd(p,prob,Tsit5())

function predict_rd()
  Flux.Tracker.collect(diffeq_rd(p,prob,Tsit5(),u0=u0))
end
loss_rd() = sum(abs2,x-1 for x in predict_rd())
loss_rd()

data = Iterators.repeated((), 50)
opt = ADAM(0.001)
cb = function ()
  display(loss_rd())
  display(plot(solve(remake(prob,u0=Flux.data(u0),p=Flux.data(p)),Tsit5(),saveat=0.1)))
end

# Display the ODE with the current parameter values.
cb()

Flux.train!(loss_rd, params(ann,p,u0), data, opt, cb = cb)
