using DiffEqFlux, Flux, DifferentialEquations
using Plots
#using ForwardDiff
using QuadGK


##  -------- Set up the problem and work out the exact solution ------------
γd = t -> cos.(2.0f0.*t) + cos.(3.0f0.*t)
#G(s) = 1/(1+s^2)
G(s) = 1/sqrt(s)
σ_f(t) = quadgk(s -> γd(t-s)*G(s),0,t)[1]
tspan = (0.0f0,12.6f0)
tsave = range(tspan[1],tspan[2],length=100)
σ_exact = vcat([0.0f0],σ_f.(tsave[2:end]))

## --- Reverse-mode AD ---
p = []
σ0 =0.0f0
u0 = Tracker.param([σ0])

f0 = Chain(Dense(2,10,tanh), Dense(10,1))
f1 = Chain(Dense(2,10,tanh), Dense(10,1))
#f0 = Chain(Dense(2,1))
#f1 = Chain(Dense(2,1))

dudt_(u::TrackedArray,p,t) = f1(Flux.Tracker.collect(vcat(u, [γd(t)])))
dudt_(u::AbstractArray,p,t) = Flux.data(f1(vcat(u,[γd(t)])))

prob = ODEProblem(dudt_,u0,tspan,p)
loss_rd() =   begin
    P_RD = vcat(Flux.Tracker.collect(diffeq_rd(p,prob,Tsit5(),u0=u0,saveat=tsave))
                  ,γd.(tsave)')
    σ_out = [f0(P_RD[:,i])[1] for i = 1:size(P_RD,2)]
    return sum( (σ_out .- σ_exact).^2 )
end

data = Iterators.repeated((), 4000)
opt = ADAM(0.015)
Flux.train!(loss_rd, params(f0,f1,p,u0), data, opt)


## -------- Evaluate against training data -------------------
display(loss_rd())
ode_solve = solve(remake(prob,u0=Flux.data(u0),p=Flux.data(p)),Tsit5())
f0_ = Flux.data(f0)
ϕ = [f0_([ode_solve(t)[1],γd(t)]).data[1] for t in tsave]
plt = plot(tsave,ϕ,m=:circle,label="NN solution",ylabel="stress",xlabel="time")
plot!(tsave,γd.(tsave),l=:dot,label="Strain rate")
display(plot!(plt,tsave,σ_exact,label="True solution"))
savefig("/Users/Dominic/Dropbox (MIT)/18.337 Final Project/FracTrainNon.pdf")



## ------------ Test against new data ----------------------
γd = t -> 1.2f0*cos.(1.0f0.*t) + 0.8f0*cos.(3.0f0.*t)
σ0 =0.0f0
σ_exact =vcat([0.0f0],σ_f.(tsave[2:end]))
prob = ODEProblem(dudt_,u0,tspan,p)
ode_solve = solve(remake(prob,u0=Flux.data(u0),p=Flux.data(p)),Tsit5())
f0_ = Flux.data(f0)
ϕ = [f0_([ode_solve(t)[1],γd(t)]).data[1] for t in tsave]
plt = plot(tsave,ϕ,m=:circle,label="NN solution",ylabel="stress",xlabel="time")
plot!(tsave,γd.(tsave),l=:dot,label="Strain rate")
display(plot!(plt,tsave,σ_exact,label="True solution"))
savefig("/Users/Dominic/Dropbox (MIT)/18.337 Final Project/FracTestNon.pdf")



## ----------------- View NN ----------------------------
ϕ_mesh = range(-0.5,0.5,length=40)
γd_mesh = range(-2.0,2.0,length=60)
f0_fun(x, y) =  f0([x,y]).data[1]
f1_fun(x, y) =  f1([x,y]).data[1]
p1 = contour(ϕ_mesh, γd_mesh, f0_fun, fill=true,c=:PRGn,
        xlabel="phi field",ylabel="strain rate",colorbar_title="f0")
p2 = contour(ϕ_mesh, γd_mesh, f1_fun, fill=true,c=:PRGn,
            xlabel="phi field",ylabel="strain rate",colorbar_title="f1")
plot(p1,p2,size=(800,300))
savefig("/Users/Dominic/Dropbox (MIT)/18.337 Final Project/NeuralNet.pdf")
