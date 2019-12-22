using DelimitedFiles
er_outfile = "/Users/Dominic/Documents/MLVE/er_data.txt"
er_dat = readdlm(er_outfile, '\t', Float32, '\n')
plt_outfile = "/Users/Dominic/Documents/MLVE/plt_data.txt"
plt_dat = readdlm(plt_outfile, '\t', Float32, '\n')

p1 = plot(er_dat[:,1],xscale=:log10,yscale=:log10,ylabel="Error",xlabel="Training steps",
        label="Training error, Neural net")
plot!(p1,er_dat[:,2],label="Testing error, Neural net")
plot!(p1,er_dat[:,3], label="Training error, linear model")
plot!(p1,er_dat[:,4],label="Testing error, linear model")

p2 = plot(plt_dat[:,1],plt_dat[:,3],m=:hexagon, lw=3, ms=3,
        label="Linear model",ylabel="stress",xlabel="time")
plot!(p2,plt_dat[:,1],plt_dat[:,2],m=:circle,lw=3,ms=3,
        label="NN solution")
plot!(p2,plt_dat[:,1],plt_dat[:,4],label="True solution",lw=3,lc=:black,
        xlims=(0,7),leg=:bottomright)

pt = plot(p1,p2,layout=(2,1),size=(600,500))

savefig(pt, "/Users/Dominic/Documents/MLVE/CombFig.pdf")
