using Plots 
using DelimitedFiles

include("model_solver.jl")

# Main simulation

params = Parameters()

@time optimal_policies, dynamics, _ = solve_dynamic_game(params, verbose = true)
reference = solve_reference_game(params)

ergodic = ergodic_distribution(dynamics.T)  # unconditional
ergodic1 = ergodic_distribution(dynamics.T1)  # type 1
ergodic2 = ergodic_distribution(dynamics.T2)  # type 2


# Plots 

# P
f1 = plot(params.rho_grid, optimal_policies.P, xlabel="rho", ylabel="Price Level P", linewidth = 2, label = "")
plot!(f1, params.rho_grid, reference.P0, xlabel="rho", ylabel="Price Level P", title="Reference Price Level vs. Reputation rho", linewidth = 2, label = "", linestyle = :dash)


# mu
f2 = plot(params.rho_grid, optimal_policies.mu1_pol, xlabel="rho", ylabel="mu", title="Optimal mus", linewidth = 2, label = "" )
plot!(f2, params.rho_grid, optimal_policies.mu2_pol, linewidth = 2, label = "")
plot!(f2, params.rho_grid, reference.mu0_1, label = "", linestyle = :dash, linewidth = 2)
plot!(f2, params.rho_grid, reference.mu0_2, label = "", linestyle = :dash, linewidth = 2)


# g 
f3 = plot(params.rho_grid, optimal_policies.g, xlabel="rho", ylabel="g", title="g Function", linewidth = 2)


# V1
f4a = plot(params.rho_grid, optimal_policies.V1, xlabel="rho", ylabel="V", title="Value Function V1", linewidth = 2, label = "V1")
plot!(f4a, params.rho_grid, reference.V1, linewidth = 2, label = "Ref V1", linestyle = :dash)


# V2
f4b = plot(params.rho_grid, optimal_policies.V2, xlabel="rho", title="Value Function V2", linewidth = 2, label = "V2")
plot!(f4b, params.rho_grid, reference.V2, linewidth = 2, label = "Ref V2", linestyle = :dash)


# V
f4c = plot(params.rho_grid, optimal_policies.V, xlabel="rho", ylabel="V", title="HH value Function V", linewidth = 2, label = "V")
plot!(f4c, params.rho_grid, reference.V, linewidth = 2, label = "Ref V", linestyle = :dash)


# Ergodic distributions
rhostep = params.rho_grid[2] - params.rho_grid[1]   # scaling the ergodic distribution by the step size in rho_grid to get a density 
f5 = plot(params.rho_grid, ergodic ./ rhostep, xlabel="rho", ylabel="Ergodic Distribution", title="Ergodic Distribution over Reputation rho", linewidth = 2, label = "unconditional")
plot!(f5, params.rho_grid, ergodic1 ./ rhostep, linewidth = 2, label = "type 1", linestyle = :dash)
plot!(f5, params.rho_grid, ergodic2 ./ rhostep, linewidth = 2, label = "type 2", linestyle = :dashdot)


# Reputation paths
paths1 = simulate_rho_paths(0.5, dynamics, params; periods = 100)
rho_path, rho_path_1, rho_path_2 = paths1
f6 = plot(rho_path, xlabel="Time Periods", ylabel="Reputation rho", title="Reputation Path over Time from mean", linewidth = 2, label = "Overall")
plot!(f6, rho_path_1, linewidth = 2, label = "Type 1", linestyle = :dash)
plot!(f6, rho_path_2, linewidth = 2, label = "Type 2", linestyle = :dashdot)


# Reputation paths from initial value 0.2
paths2 = simulate_rho_paths(0.2, dynamics, params; periods = 100)
rho_path, rho_path_1, rho_path_2 = paths2
f7 = plot(rho_path, xlabel="Time Periods", ylabel="Reputation rho", title="Reputation Path over Time from low reputation", linewidth = 2, label = "Overall")
plot!(f7, rho_path_1, linewidth = 2, label = "Type 1", linestyle = :dash)
plot!(f7, rho_path_2, linewidth = 2, label = "Type 2", linestyle = :dashdot)


# EXPORTING

mkpath("results")

# displaying and saving figures
for (i, f) in enumerate([f1, f2, f3, f4a, f4b, f4c, f5, f6, f7])
    display(f)
    savefig(f, joinpath("results", string("figure$i", ".png")))
end


# Exporting data to CSV files for plotting in Tikz 

toexport =(
    params.rho_grid,
    optimal_policies.P,
    reference.P0,
    optimal_policies.mu1_pol,
    reference.mu0_1,
    optimal_policies.mu2_pol,
    reference.mu0_2,
    optimal_policies.V1,
    reference.V1,
    optimal_policies.V2,
    reference.V2,
    optimal_policies.V,
    reference.V,
    ergodic ./ rhostep)

writedlm(joinpath("results", "rhoaxisplots.csv"), [ [a...]  for a in zip(toexport...)],  ',')
writedlm(joinpath("results", "pathplots.csv"), [ [a...]  for a in zip(paths1..., paths2...)],  ',')
