using DiffEqFlux, Flux, DifferentialEquations, Plots
using Statistics: mean
## --- Set up parameters ---

u0 = Float32[0.; 0.]
tspan = (-15.0f0,15.0f0)
datasize = 30
t = range(tspan[1],tspan[2],length=datasize)
#ann = Chain(Dense(2,50,tanh), Dense(50,1))
#tst = (dim,b)->[-1.0;-0.95]
#d = Dense(2,1,initW = (dims...)->[-1.0f0  -0.95f0],initb = (dims...)->[0.0f0])
d = Dense(2,1)

ann = Chain(d)
p = param(DiffEqFlux.destructure(ann))

## ---- Create true ode data ----
#γd(t) = exp(-0.05*t^2)
γdd(t) = -0.1*t*exp(-0.05*t^2)
function trueODEfunc(du,u,p,t)
    du[1] = -1.1*u[1] - u[2]
    du[2] = γdd(t)
end
prob = ODEProblem(trueODEfunc,u0,tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))

## ---- Set up Neural ode ----
function dudt_(du,u,p,t)
    du[1] = DiffEqFlux.restructure(ann,p)(u)[1]
    du[2] = γdd(t)
end
prob = ODEProblem(dudt_,u0,tspan,p)
#diffeq_adjoint(p,prob,Tsit5(),u0=u0,abstol=1e-8,reltol=1e-6)

function predict_adjoint()
  diffeq_adjoint(p,prob,Tsit5(),u0=u0,saveat=t,
    abstol=1e-9,reltol=1e-9)
end

loss_adjoint() = mean(abs2,ode_data[1,:] .- predict_adjoint()[1,:])

data = Iterators.repeated((), 100)
opt = Descent(0.1)
cb = function ()
  display(loss_adjoint())
#  cur_pred = Flux.data(predict_adjoint())
#  pl = scatter(t,ode_data[1,:],label="data")
#  scatter!(pl,t,cur_pred[1,:],label="prediction")
#  display(plot(pl))
end

# Display the ODE with the current parameter values.
cb()
ps = Flux.params(p,u0)
Flux.train!(loss_adjoint, ps, data, opt, cb = cb)
