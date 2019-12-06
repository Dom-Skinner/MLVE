using DiffEqFlux, Flux, DifferentialEquations
using Plots
using QuadGK

function mode_loss(f0,f1,γd,G)
    σ_f(t) = quadgk(s -> γd(t-s)*G(s),0,t)[1]
    tspan = (0.0f0,6.2831f0)
    tsave = range(tspan[1],tspan[2],length=100)
    σ_exact = vcat([0.0f0],σ_f.(tsave[2:end]))
    p = []
    σ0 =0.0f0
    u0 = Tracker.param([σ0])

    dudt_(u::TrackedArray,p,t) = f1(Flux.Tracker.collect(vcat(u, [γd(t)])))
    dudt_(u::AbstractArray,p,t) = Flux.data(f1(vcat(u,[γd(t)])))

    prob = ODEProblem(dudt_,u0,tspan,p)
    loss_rd() =   begin
        P_RD = vcat(Flux.Tracker.collect(diffeq_rd(p,prob,Tsit5(),u0=u0,saveat=tsave))
                      ,γd.(tsave)')
        σ_out = [f0(P_RD[:,i])[1] for i = 1:size(P_RD,2)]
        return sum( (σ_out .- σ_exact).^2 )
    end
    return loss_rd
end

function test_NN(γd,G,f0,f1)
    tspan = (0.0f0,10.0f0)
    tsave = range(tspan[1],tspan[2],length=100)
    σ_f(t) = quadgk(s -> γd(t-s)*G(s),0,t)[1]
    σ_exact = vcat([0.0f0],σ_f.(tsave[2:end]))
    dudt_(u::AbstractArray,p,t) = Flux.data(f1(vcat(u,[γd(t)])))
    prob = ODEProblem(dudt_,[0.0f0],tspan,p)
    ode_solve = solve(prob,u0=[0.0f0],Tsit5())
    f0_ = Flux.data(f0)
    σ_approx = [f0_([ode_solve(t)[1],γd(t)]).data[1] for t in tsave]

    return tsave, σ_approx, σ_exact
end
function test_err(γd,G,f0,f1)
    tsave, σ_approx, σ_exact = test_NN(γd,G,f0,f1)
    return sum( (σ_approx .- σ_exact).^2 )
end
##  -------- Set up the problem and work out the exact solution ------------
#G(s) = 1/(1+s^2)
G(s) = 1/sqrt(s)
## --- Reverse-mode AD ---

f0_n = Chain(Dense(2,10,tanh), Dense(10,1))
f1_n = Chain(Dense(2,10,tanh), Dense(10,1))
f0_l = Chain(Dense(2,1))
f1_l = Chain(Dense(2,1))

t_loss(f0,f1) =  sum(mode_loss(f0,f1,t -> cos.(ω.*t),G)() for ω in 0.0f0:5.0f0)
data = Iterators.repeated((), 2000)
opt = ADAM(0.015)
p = []
γd_test = t -> 0.5*cos.(1.5f0.*t) + 0.5*cos.(3.5f0.*t)
err_l = []
err_n = []
cb(err,f0,f1) = push!(err,[t_loss(f0,f1).data,test_err(γd_test,G,f0,f1)])

Flux.train!(() -> t_loss(f0_n,f1_n), params(f0_n,f1_n,p), data, opt,cb= () -> cb(err_n,f0_n,f1_n))
Flux.train!(() -> t_loss(f0_l,f1_l), params(f0_l,f1_l,p), data, opt,cb= () -> cb(err_l,f0_l,f1_l))

Er = (i,er) -> [e[i] for e in er]
plot(Er(1,err_n),xscale=:log10,yscale=:log10,ylabel="Error",xlabel="Training steps",
        label="Training error, Neural net")
plot!(Er(2,err_n),label="Testing error, Neural net")
plot!(Er(1,err_l), label="Training error, linear model")
plot!(Er(2,err_l),label="Testing error, linear model")


tsave, σ_approx_l, σ_exact = test_NN(γd,G,f0_l,f1_l)
tsave, σ_approx_n, σ_exact = test_NN(γd,G,f0_n,f1_n)
#plot!([1,length(err_n)],sum((σ_approx_n .- σ_exact).^2)*ones(2),l=:dot,label="Neural net (Test data)")
#plot!([1,length(err_l)],sum((σ_approx_l .- σ_exact).^2)*ones(2),l=:dot,label="Linear model (Test data)")
savefig("/Users/Dominic/Dropbox (MIT)/18.337 Final Project/FracComp.pdf")


## -------- Evaluate against training data -------------------
display(t_loss())
γd = t -> cos.(1.0f0.*t) + cos.(3.0f0.*t)
tsave, σ_approx, σ_exact = test_NN(γd,G,f0_l,f1_l)
println(sum((σ_approx .- σ_exact).^2))
plt = plot(tsave,σ_approx,m=:circle,label="NN solution",ylabel="stress",xlabel="time")
plot!(tsave,γd.(tsave),l=:dot,label="Strain rate")
display(plot!(plt,tsave,σ_exact,label="True solution"))
savefig("/Users/Dominic/Dropbox (MIT)/18.337 Final Project/FracTestNon.pdf")



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
ϕ_mesh = range(-2.5,2.5,length=40)
γd_mesh = range(-9.0,9.0,length=60)
f0_fun(x, y) =  f0_n([x,y]).data[1]
f1_fun(x, y) =  f1_n([x,y]).data[1]
p1 = contour(ϕ_mesh, γd_mesh, f0_fun, fill=true,c=:PRGn,
        xlabel="phi field",ylabel="strain rate",colorbar_title="f0")
p2 = contour(ϕ_mesh, γd_mesh, f1_fun, fill=true,c=:PRGn,
            xlabel="phi field",ylabel="strain rate",colorbar_title="f1")
plot(p1,p2,size=(800,300))
savefig("/Users/Dominic/Dropbox (MIT)/18.337 Final Project/NeuralNet.pdf")
