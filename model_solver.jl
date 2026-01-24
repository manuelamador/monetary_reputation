using Roots: find_zero
using LinearAlgebra: dot 
using OhMyThreads: @tasks, tmapreduce


# Structs to hold parameters of the model

# Distribution struct: holds grids and pdfs for two types
struct Distribution{T} 
    mua_grid::Vector{T}
    pdf_A::Vector{T} 
    pdf_B::Vector{T} 
end 
get_distribution(a::Distribution) = a

"""
    TruncatedNormals(; max, points, meanA, meanB, sigmaA=meanA, sigmaB=meanB)

Create a `Distribution` with truncated normal PDFs for types A and B.

# Arguments
- `max`: Upper bound of the support (lower bound is 0)
- `points`: Number of grid points
- `meanA`, `meanB`: Means for type A and B distributions
- `sigmaA`, `sigmaB`: Standard deviations (default to means)

# Returns
- `Distribution` struct with discretized and normalized PDFs
"""
function TruncatedNormals(; max, points, meanA, meanB, sigmaA = meanA, sigmaB = meanB)
    mu_grid = range(0.0, stop = max, length = points)
    # Precompute density components to avoid repeated exp() in hot loops
    pdf_A = [1 / (sigmaA * sqrt(2π)) * exp(-0.5 * ((mu - meanA) / sigmaA)^2) for mu in mu_grid]
    pdf_B = [1 / (sigmaB * sqrt(2π)) * exp(-0.5 * ((mu - meanB) / sigmaB)^2) for mu in mu_grid]
    # normalize to ensure they integrate to 1 over the grid
    pdf_A ./= sum(pdf_A)
    pdf_B ./= sum(pdf_B)
    return Distribution(collect(mu_grid), pdf_A, pdf_B)    
end 

"""
    TruncatedExponentials(; max, points, meanA, meanB)

Create a `Distribution` with truncated exponential PDFs for types A and B.

# Arguments
- `max`: Upper bound of the support (lower bound is 0)
- `points`: Number of grid points
- `meanA`, `meanB`: Means for type A and B distributions (λ = 1/mean)

# Returns
- `Distribution` struct with discretized and normalized PDFs
"""
function TruncatedExponentials(; max, points, meanA, meanB)
    mua_grid = range(0.0, stop = max, length = points)

    lambdaA = 1 / meanA
    lambdaB = 1 / meanB

    # Precompute density components to avoid repeated exp() in hot loops
    pdf_A = [lambdaA * exp(-lambdaA * mua) for mua in mua_grid]
    pdf_B = [lambdaB * exp(-lambdaB * mua) for mua in mua_grid]
   
    # normalize to ensure they integrate to 1 over the grid
    pdf_A ./= sum(pdf_A)
    pdf_B ./= sum(pdf_B)

    return Distribution(collect(mua_grid), pdf_A, pdf_B)
end

# f: Distribution function evaluated at grid index mua_i with mixture parameter mu
# Returns weighted average of two exponential distributions
f(mua_i, mu, params) = mu * get_distribution(params).pdf_A[mua_i] + (1 - mu) * get_distribution(params).pdf_B[mua_i]

# df: Derivative of distribution function with respect to mu
# Used in FOC for optimal policy
df(mua_i, params) = get_distribution(params).pdf_A[mua_i] - get_distribution(params).pdf_B[mua_i]

# f_given_policies: Distribution function given current policy functions
# Computes the implied distribution when types follow their optimal policies
function f_given_policies(mua_i,  rho_i, params, policies)
    (; mu1_pol, mu2_pol) = policies
    rho_grid = get_rho_grid(params)

    rho = rho_grid[rho_i]
    return rho * f(mua_i, mu1_pol[rho_i], params) + (1 - rho) * f(mua_i, mu2_pol[rho_i], params)
end


# Utility parameters struct
struct UtilityParameters{T}
    epsilon::T
    delta::T
    sigma::T
    beta::T
    alpha::T
    beta1::T 
    beta2::T
    rho_grid::Vector{T}
end
get_parameters(a::UtilityParameters) = a

# Inflation costs struct
struct InflationCosts{T, F1, F2, F3}
    gamma1::T
    gamma2::T
    h::F1 
    hprime::F2
    inv_hprime::F3
end
get_inflation_costs(a::InflationCosts) = a

# Model parameters struct: holds all model parameters
struct ModelParameters{T, F1, F2, F3}
    distribution::Distribution{T}
    utility::UtilityParameters{T}
    inflation_costs::InflationCosts{T, F1, F2, F3}
end 

get_distribution(params) = params.distribution
get_parameters(params) = params.utility
get_inflation_costs(params) = params.inflation_costs
get_rho_grid(params) = get_parameters(params).rho_grid


"""
    ModelParameters(; kwargs...)

Construct model parameters for the monetary policy reputation game.

# Keyword Arguments
- `gammas`: Either `(; π_target_1, π_target_2)` for inflation targets, or `(; gamma1, gamma2)` for direct cost parameters
- `epsilon`: Probability type 2 becomes type 1 (default: 0.02)
- `delta`: Probability type 1 becomes type 2 (default: 0.02)
- `sigma`: Elasticity of substitution (default: 5.0)
- `beta`: Discount factor (default: 0.99)
- `alpha`: Utility parameter (default: 1.0)
- `rho_points`: Number of reputation grid points (default: 200)
- `rho_grid`: Custom reputation grid (default: range from epsilon to 1-delta)
- `distribution`: Distribution struct (default: TruncatedExponentials)
- `h`, `hprime`, `inv_hprime`: Inflation cost function and its derivative/inverse

# Returns
- `ModelParameters` struct containing all model parameters

# Example
```julia
params = ModelParameters(gammas = (; π_target_1 = 0.02, π_target_2 = 0.04))
```
"""
function ModelParameters(;
    gammas = (;  π_target_1 = 0.02, π_target_2 = 0.03),
    epsilon = 0.02, delta = 0.02, sigma = 5.0, beta = 0.99, alpha = 1.0,
    rho_points = 200,
    rho_grid = range(epsilon, stop = 1 - delta, length = rho_points),
    distribution = nothing, # defaults to exponential if not provided
    h = (mu) -> 1/2 * mu^2,
    hprime = (mu) -> mu,
    inv_hprime = (x) -> x
) 
 
    beta1 = beta * (1 - delta)
    beta2 = beta * (1 - epsilon)

    utility = UtilityParameters(
        epsilon,
        delta,
        sigma,
        beta,
        alpha,
        beta1,
        beta2,
        collect(rho_grid)
    )

    if distribution === nothing
        distribution = TruncatedExponentials(
            max = 0.5,
            points = 2000,
            meanA = 0.04,
            meanB = 0.01
        )
    end

    if haskey(gammas, :π_target_1) && haskey(gammas, :π_target_2)
        gamma1, gamma2 = obtain_gammas_given_target(gammas.π_target_1, gammas.π_target_2, (; utility, distribution), hprime)
    else
        (; gamma1, gamma2) = gammas
    end

    inflation_costs = InflationCosts(
        gamma1,
        gamma2,
        h,
        hprime,
        inv_hprime
    )

    return ModelParameters(
        distribution,
        utility,
        inflation_costs)
end


######################################################################################

# Policies struct: container for policy and value functions
# - mu1_pol, mu2_pol: optimal money injection policies for each type
# - P: price level, V: household value function
# - V1, V2: value functions for type 1 and type 2
# - c, Pc: consumption and price times consumption
# - g: expectation used in pricing equation
struct Policies{V, V1, M}
    mu1_pol::V
    mu2_pol::V
    P::V1
    V::V1
    V1::V1
    V2::V1
    c::M
    Pc::M
    g::V1
end 

# Constructor for Policies: initializes with default values
# - Policy functions start at 0
# - Price level starts at 1
# - Value functions start at -alpha
# - Consumption matrices start at 1
function Policies(params) 
    rho_grid = get_rho_grid(params)
    alpha = get_parameters(params).alpha
    mua_grid = get_distribution(params).mua_grid

    return Policies(
    [0.0 for _ in rho_grid],
    [0.0 for _ in rho_grid],
    [1.0 for _ in rho_grid],
    [- alpha for _ in rho_grid],
    [- alpha for _ in rho_grid],
    [- alpha for _ in rho_grid],
    ones(length(mua_grid), length(rho_grid)), 
    ones(length(mua_grid), length(rho_grid)),  
    [1.0 for _ in rho_grid]
)
end 




#######################################################################################
# Functions to obtain gamma1 and gamma2 given target inflation rates
# for the case where types are permanent and known. 

# Find the mu1 and mu2 that yield the target inflation rates:
# E[mua] = mu * E[mua| A] + (1 - mu) * E[mua| B] = target
function obtain_mu_given_targets(π_target_1, π_target_2, params) 
    (; pdf_A, pdf_B, mua_grid) = get_distribution(params)

    EmuA = sum(mua_grid[i] * pdf_A[i] for i in eachindex(mua_grid))
    EmuB = sum(mua_grid[i] * pdf_B[i] for i in eachindex(mua_grid))

    mu1  = (π_target_1 - EmuB) / (EmuA - EmuB)
    mu2  = (π_target_2 - EmuB) / (EmuA - EmuB)
    return mu1, mu2
end 


# Function to obtain gamma1 and gamma2 given mu1 and mu2
function obtain_gammas_given_target(π_target_1, π_target_2, params, hprime)
    mu1, mu2 = obtain_mu_given_targets(π_target_1, π_target_2, params) 

    gamma1 = mu_foc_lhs(mu1, params) / hprime(mu1) 
    gamma2 = mu_foc_lhs(mu2, params) / hprime(mu2)

    return gamma1, gamma2
end 



#####################################################################################
# Solving the dynamic game

#####################################################################################
# Belief updating functions

# rhohat: Posterior belief using Bayes' rule
# Updates belief about type after observing money shock
# Returns probability that policymaker is type 1 given the shock
function rhohat(mua_i, rho_i, params, policies) 
    (; mu1_pol, mu2_pol) = policies
    rho = get_rho_grid(params)[rho_i] 
    return rho * f(mua_i, mu1_pol[rho_i], params) / (
        rho * f(mua_i, mu1_pol[rho_i], params) + (1 - rho) * f(mua_i, mu2_pol[rho_i], params)
    )
end


# rhoplus: Updated belief accounting for type transitions
# Incorporates probability delta that type 1 becomes type 2
# and probability epsilon that type 2 becomes type 1
function rhoplus(mua_i, rho_i, params, policies) 
    (; delta, epsilon) = get_parameters(params)
    return (1 - delta) * rhohat(mua_i, rho_i, params, policies) + epsilon * (1 - rhohat(mua_i, rho_i, params, policies))
end


#####################################################################################
# Equilibrium object computation

# update_g!: Compute the g function (expectation of 1/c)
# Used in the consumption optimality condition
# Modifies g in-place
function update_g!(g, params, policies) 
    (; mua_grid) = get_distribution(params)
    @tasks for rho_i in eachindex(g)
        g[rho_i] = 0.0 
        for mua_i in eachindex(mua_grid)
            g[rho_i] += (1 / policies.c[mua_i, rho_i]) * f_given_policies(mua_i, rho_i, params, policies)
        end
    end 
end


# update_Pc!: Compute P times c using the consumption optimality condition
# Households choose consumption to satisfy Euler equation
# Result capped at (1 + mua) to ensure c ≤ (1 + mua) / P
# Modifies new_policies.Pc in-place
function update_Pc!(new_policies, policies, params) 
    rho_grid = get_rho_grid(params)
    (; mua_grid) = get_distribution(params)
    (; beta) = get_parameters(params)
    (; P, g) = policies
    (; Pc) = new_policies

    @tasks for rho_i in eachindex(rho_grid)
        rho = rho_grid[rho_i]
        for mua_i in eachindex(mua_grid)
            mua = mua_grid[mua_i]
            Pc[mua_i, rho_i] = (1 + mua) * min(linear_interp(rho_grid, P, rhoplus(mua_i, rho_i, params, policies)) / (beta * linear_interp(rho_grid, g, rho)), 1)
        end
    end
end


# update_P!: Compute price level P using the pricing optimality condition
# Firms choose prices optimally given demand and expectations
# Modifies new_policies.P in-place
function update_P!(new_policies, policies, params) 
    rho_grid = get_rho_grid(params)
    (; mua_grid) = get_distribution(params)
    (; alpha, beta, sigma) = get_parameters(params)
    (; g, P) = policies
    (; Pc) = new_policies
    newP = new_policies.P

    @tasks for j in eachindex(rho_grid)
        term1 = 0.0 
        term2 = 0.0 
        for i in eachindex(mua_grid)
            mua = mua_grid[i]
            rho_prime = rhoplus(i, j, params, policies)
            term1 += beta * (1- sigma) * (1/ (1 + mua)) * linear_interp(rho_grid, g, rho_prime) / linear_interp(rho_grid, P, rho_prime) * Pc[i, j] * f_given_policies(i, j, params, policies) 

            term2 += alpha * sigma * Pc[i, j] * f_given_policies(i, j, params, policies)
        end
        newP[j] = - term2 / term1
    end
end


# update_c!: Construct consumption function from Pc and P
# Simple division: c = Pc / P
# Modifies new_policies.c in-place
function update_c!(new_policies, params)
    (; mua_grid) = get_distribution(params)
    rho_grid = get_rho_grid(params)
    (; c, Pc, P) = new_policies

    @tasks for j in eachindex(rho_grid)
        for i in eachindex(mua_grid)
            c[i, j] = Pc[i, j] / P[j]
        end 
    end 
end 


# update_all!: Update all equilibrium objects given old policies
# Sequentially computes g, Pc, P, and c
# Does not update optimal mu policies (see optimize_policies!)
function update_all!(new_policies, old_policies, params)
    update_g!(old_policies.g, params, old_policies)
    update_Pc!(new_policies, old_policies, params)
    update_P!(new_policies, old_policies, params)
    update_c!(new_policies, params)
    
end 


# optimize_policies!: Compute optimal mu policies and value functions
# For each reputation level:
#   1. Compute FOC for optimal mu1 and mu2 using foc
#   2. Integrate to get value functions V1, V2, V for each type
#   3. Account for cost of money injection h(mu)
# Modifies new_policies.mu1_pol, mu2_pol, V1, V2, V in-place
function optimize_policies!(new_policies, old_policies, params)
    rho_grid = get_rho_grid(params)
    (; mua_grid) = get_distribution(params) 
    (; alpha, beta, beta1, beta2) = get_parameters(params)
    (; gamma1, gamma2, h, inv_hprime) = get_inflation_costs(params)

    (; V, V1, V2) = old_policies
    (; c) = new_policies
    
    @tasks for rho_i in eachindex(rho_grid)
     
        # using the FOC to find optimal mu for each type 
        dvalue1 = 0.0
        dvalue2 = 0.0
        # integrating the derivative of the payoffs
        for mua_i in eachindex(mua_grid) 
            rho_prime = rhoplus(mua_i, rho_i, params, old_policies)
            dvalue1 += ((1 - beta1) * (log(c[mua_i, rho_i]) - alpha * c[mua_i, rho_i]) + beta1 * linear_interp(rho_grid, V1, rho_prime)) * df(mua_i, params)
            dvalue2 += ((1 - beta2) * (log(c[mua_i, rho_i]) - alpha * c[mua_i, rho_i]) + beta2 * linear_interp(rho_grid, V2, rho_prime)) * df(mua_i, params)
        end 
        dvalue1 = dvalue1 / ((1- beta1) * gamma1)   
        dvalue2 = dvalue2 / ((1- beta2) * gamma2) 
        
        best_mu1 = clamp(inv_hprime(dvalue1), 0.0, 1.0)
        best_mu2 = clamp(inv_hprime(dvalue2), 0.0, 1.0)

        new_policies.mu1_pol[rho_i] = best_mu1
        new_policies.mu2_pol[rho_i] = best_mu2

        # computing the value function at the optimal mu
        value1 = 0.0
        value2 = 0.0
        value = 0.0
        # integrating the payoffs
        for mua_i in eachindex(mua_grid) 
            rho_prime = rhoplus(mua_i, rho_i, params, new_policies)
            value += ((1 - beta) * (log(c[mua_i, rho_i]) - alpha * c[mua_i, rho_i]) + beta * linear_interp(rho_grid, V, rho_prime)) * f_given_policies(mua_i, rho_i, params, new_policies)
            value1 += ((1 - beta1) * (log(c[mua_i, rho_i]) - alpha * c[mua_i, rho_i]) + beta1 * linear_interp(rho_grid, V1, rho_prime)) * f(mua_i, best_mu1, params)
            value2 += ((1 - beta2) * (log(c[mua_i, rho_i]) - alpha * c[mua_i, rho_i]) + beta2 * linear_interp(rho_grid, V2, rho_prime)) * f(mua_i, best_mu2, params)
        end 
        best_value1 = value1 - ((1- beta1) * gamma1) * h(best_mu1)  
        best_value2 = value2 - ((1- beta2) * gamma2) * h(best_mu2)

        new_policies.V1[rho_i] = best_value1
        new_policies.V2[rho_i] = best_value2
        new_policies.V[rho_i] = value
    end
end 


#####################################################################################
# Main solver iteration

# iterate_until_convergence: Main fixed-point iteration loop
# Iteratively updates policies and values until convergence
# Uses damping (gain parameter) to facilitate convergence
# Prints progress every 10 iterations
# Returns converged policies
function iterate_until_convergence(params; max_iters = 2_000, tol = 1e-6, 
            old_policies = Policies(params), new_policies = Policies(params), 
            gain = 0.5, verbose = true, printevery = 10)

    @assert 0.0 < gain ≤ 1.0
    verbose && println("Iterating value and policy functions ...")

    iter = 1
    diff = 0.0
    while iter <= max_iters
        update_all!(new_policies, old_policies, params)
        optimize_policies!(new_policies, old_policies, params)

        diff = get_distance(new_policies, old_policies)
        verbose && (iter % printevery == 0) && println("    Iteration $iter, diff = $diff")
        (diff < tol) && break

        if gain < 1.0
            # to facilitate convergence, do a convex combination of old and new values
            new_policies.V1 .= gain * new_policies.V1 .+ (1 - gain) * old_policies.V1
            new_policies.V2 .= gain * new_policies.V2 .+ (1 - gain) * old_policies.V2
            new_policies.P .= gain * new_policies.P .+ (1 - gain) * old_policies.P
        end

        old_policies, new_policies = new_policies, old_policies
        iter += 1
    end
    iter == max_iters && println("WARNING: Maximum iterations reached without convergence.")
    verbose && println("... done.")
    return new_policies, old_policies, diff
end


#####################################################################################
# Dynamics and transition matrices

"""
    build_transition_matrices(params, policies)

Build transition matrices for reputation dynamics given converged policies.

For each current reputation level and money shock realization, computes the
next period's reputation using Bayesian updating and type transition probabilities.
Probability mass is distributed using linear interpolation on the reputation grid.

# Arguments
- `params::ModelParameters`: Model parameters
- `policies::Policies`: Converged policy functions

# Returns
- `T`: Overall transition matrix (N×N)
- `T1`: Transition matrix conditional on type 1
- `T2`: Transition matrix conditional on type 2

Where N = length of reputation grid. Rows sum to 1.
"""
function build_transition_matrices(params, policies)
    rho_grid = get_rho_grid(params)
    (; mua_grid) = get_distribution(params)
    (; mu1_pol, mu2_pol) = policies

    N = length(rho_grid)
    T = zeros(N, N)
    T1 = zeros(N, N)
    T2 = zeros(N, N)

    @tasks for rho_i in eachindex(rho_grid)
        for mua_i in eachindex(mua_grid)
            rho_prime = rhoplus(mua_i, rho_i, params, policies)
            # use the linear interpolation search code to fill in the transition matrix
            i = searchsortedlast(rho_grid, rho_prime)
            i = clamp(i, 1, length(rho_grid)-1)
            t = (rho_prime - rho_grid[i]) / (rho_grid[i+1] - rho_grid[i])
            weight = f_given_policies(mua_i, rho_i, params, policies) 
            weight1 = f(mua_i, mu1_pol[rho_i], params)
            weight2 = f(mua_i, mu2_pol[rho_i], params)

            T[rho_i, i] += (1 - t) * weight 
            T[rho_i, i+1] += t * weight 
            T1[rho_i, i] += (1 - t) * weight1
            T1[rho_i, i+1] += t * weight1
            T2[rho_i, i] += (1 - t) * weight2
            T2[rho_i, i+1] += t * weight2
        end
        T[rho_i, :] ./= sum(T[rho_i, :])
        T1[rho_i, :] ./= sum(T1[rho_i, :])
        T2[rho_i, :] ./= sum(T2[rho_i, :])
    end
    return T, T1, T2
end


"""
    solve_dynamic_game(params; kwargs...)

Solve the dynamic monetary policy game with reputation effects.

Iteratively computes optimal policies for both policymaker types and the resulting
equilibrium prices, consumption, and value functions. Also builds transition matrices
for reputation dynamics.

# Arguments
- `params::ModelParameters`: Model parameters

# Keyword Arguments
- `policies_1`, `policies_2`: Initial policy guesses (default: `Policies(params)`)
- `max_iters`: Maximum iterations (default: 5000)
- `tol`: Convergence tolerance (default: 1e-6)
- `gain`: Damping parameter for convergence (default: 0.5)
- `verbose`: Print progress (default: true)

# Returns
- `optimal_policies::Policies`: Converged policy and value functions
- `dynamics::NamedTuple`: Transition matrices `(T, T1, T2)` for reputation dynamics
- `diff::Float64`: Final convergence metric

# Example
```julia
params = ModelParameters()
policies, dynamics, diff = solve_dynamic_game(params)
```
"""
function solve_dynamic_game(params; policies_1 = Policies(params), policies_2 = Policies(params), max_iters = 5_000, tol = 1e-6, gain = 0.5, verbose = true, printevery = 10)
    optimal_policies, _, diff = iterate_until_convergence(params; old_policies = policies_1, new_policies = policies_2, max_iters, tol, gain, verbose, printevery)
    T, T1, T2 = build_transition_matrices(params, optimal_policies)
    return optimal_policies, (; T, T1, T2), diff
end


"""
    simulate_rho_paths(rho_init, dynamics, params; periods=100)

Simulate mean reputation paths over time starting from initial reputation `rho_init`.

Evolves the reputation distribution forward using the transition matrices,
computing expected reputation paths for:
- Overall economy (mixing both types according to current ρ)
- Conditional on type 1
- Conditional on type 2

# Arguments
- `rho_init`: Initial reputation level
- `dynamics`: Named tuple with transition matrices `(T, T1, T2)` from `solve_dynamic_game`
- `params::ModelParameters`: Model parameters

# Keyword Arguments
- `periods`: Number of periods to simulate (default: 100)

# Returns
- `rho_path`: Overall expected reputation path (length `periods+1`)
- `rho_path_1`: Expected path conditional on type 1
- `rho_path_2`: Expected path conditional on type 2
"""
function simulate_rho_paths(rho_init, dynamics, params; periods = 100)
    rho_grid = get_rho_grid(params)

    i = searchsortedlast(rho_grid, rho_init)
    i = clamp(i, 1, length(rho_grid))

    rho_current = rho_grid[i]
    rho_path = [rho_current]
    rho_path_1 = [rho_current]
    rho_path_2 = [rho_current]

    rho_vector = zero(rho_grid)
    rho_vector[i] = 1.0

    rho_vector_1 = zero(rho_grid)
    rho_vector_1[i] = 1.0

    rho_vector_2 = zero(rho_grid)
    rho_vector_2[i] = 1.0

    for _ in 1:periods
        rho_vector = dynamics.T' * rho_vector
        rho_vector_1 = dynamics.T1' * rho_vector_1
        rho_vector_2 = dynamics.T2' * rho_vector_2

        rho_next = dot(rho_grid, rho_vector)
        rho_next_1 = dot(rho_grid, rho_vector_1)
        rho_next_2 = dot(rho_grid, rho_vector_2)

        push!(rho_path, rho_next)
        push!(rho_path_1, rho_next_1)
        push!(rho_path_2, rho_next_2)
    end

    return rho_path, rho_path_1, rho_path_2
end


#####################################################################################
# Solving the reference game (without reputation effects)

# expected_one_plus_mua: Compute E[1 + μᵃ] given policy mu
# Expected value of (1 + money shock) under distribution f(·, mu)
function expected_one_plus_mua(params, mu)
    (; mua_grid) = get_distribution(params)
    val = 0.0 
    for Emua_i in eachindex(mua_grid)
        Emua = mua_grid[Emua_i]
        val += (1 + Emua) * f(Emua_i, mu, params)
    end
    return val 
end

# expected_one_over_one_plus_mua: Compute E[1/(1 + μᵃ)] given policy mu
# Expected value of inverse of (1 + money shock)
function expected_one_over_one_plus_mua(params, mu)
    (; mua_grid) = get_distribution(params)
    val = 0.0 
    for Emua_i in eachindex(mua_grid)
        Emua = mua_grid[Emua_i]
        val += 1/(1 + Emua) * f(Emua_i, mu, params)
    end
    return val 
end

# compute_consumption_scaler: Helper function for reference game
# Computes the constant part of consumption as function of mu
# Used in period 1 steady state calculation
function compute_consumption_scaler(params, mu)
    (; alpha, sigma, beta) = get_parameters(params)

    return 1/alpha * (sigma - 1)/sigma * beta * expected_one_over_one_plus_mua(params, mu) / expected_one_plus_mua(params, mu) 
end 

# mu_foc_lhs: the lhs of the FOC for optimal mu in reference game (period 1)
# Returns derivative of objective with respect to mu without the inflation costs
# Used with root finder to find optimal policy
function mu_foc_lhs(mu, params)
    (; mua_grid) = get_distribution(params)
    (; alpha) = get_parameters(params)

    value = 0.0 
    part = compute_consumption_scaler(params, mu)
    for mua_i in eachindex(mua_grid)
        mua = mua_grid[mua_i]
        cons = part * (1 + mua)
        value += (log(cons) - alpha * cons) * df(mua_i, params)
    end
    return value 
end 


# Elogcminusalphac: Compute expected utility E[log(c) - alpha * c] given policy mu
function Elogcminusalphac(mu, params)
    (; mua_grid) = get_distribution(params)
    (; alpha) = get_parameters(params)

    value = 0.0 
    part = compute_consumption_scaler(params, mu)
    for mua_i in eachindex(mua_grid)
        mua = mua_grid[mua_i]
        cons = part * (1 + mua)
        value += (log(cons) - alpha * cons) * f(mua_i, mu, params)
    end
    return value 
end 


# solve_reference_period1: Solve for period 1 equilibrium in reference game
# Finds optimal mu1, mu2 and corresponding prices P1, P2
# Uses root finding on FOC for each type
# Returns named tuple (mu1, mu2, P1, P2)
function solve_reference_period1(params)   
    (; gamma1, gamma2, hprime) = get_inflation_costs(params)
    mu1 = find_zero(x -> mu_foc_lhs(x, params) - gamma1 * hprime(x), 0.5)
    mu2 = find_zero(x -> mu_foc_lhs(x, params) - gamma2 * hprime(x), 0.5)
    P1 = 1 / compute_consumption_scaler(params, mu1)
    P2 = 1 / compute_consumption_scaler(params, mu2)
    return (; mu1, mu2, P1, P2) 
end 

# compute_mu0s: Compute period 0 optimal policies given price P
# Solves for mu that satisfies FOC given P
# Returns mu01 (type 1) and mu02 (type 2)
function compute_mu0s(P, params)
    (; alpha) = get_parameters(params)
    (; mua_grid) = get_distribution(params)
    (; gamma1, gamma2, inv_hprime) = get_inflation_costs(params)

    val = 0.0 
    for mua_i in eachindex(mua_grid)
        mua = mua_grid[mua_i] 
        val += (log(1 + mua) - alpha * (1 + mua) / P) * df(mua_i, params)
    end
    mu01 = inv_hprime(val / gamma1)
    mu02 = inv_hprime(val / gamma2)
    return mu01, mu02
end 

# period0_pricing_residual: Pricing equation for period 0 in reference game
# Given mu1, mu2 (period 1 policies), rho (reputation):
#   Given P0 obtain mu0_1 and mu0_2
#   Given mu01 and mu02,  computes equilibrium P0' from pricing FOC
#   Returns residual P0' - P0 (should be zero at equilibrium)
function period0_pricing_residual(P0, mu1, mu2, rho, params)
    (; alpha, sigma, beta) = get_parameters(params)
    (; mua_grid) = get_distribution(params)
    
    mu0_1, mu0_2 = compute_mu0s(P0, params)
    num_term1 = 0.0 
    num_term2 = 0.0
    den_term1 = 0.0 
    den_term2 = 0.0 

    for mua_i in eachindex(mua_grid)
        mua = mua_grid[mua_i]
        num_term1 += (1 + mua) * f(mua_i, mu0_1, params)
        num_term2 += (1 + mua) * f(mua_i, mu0_2, params)
        den_term1 += 1/(1 + mua) * f(mua_i, mu1, params)
        den_term2 += 1/(1 + mua) * f(mua_i, mu2, params)
    end
    return alpha * sigma / (sigma - 1) * 1 / beta *  (rho * num_term1 + (1 - rho) * num_term2) / (rho * den_term1 + (1 - rho) * den_term2) - P0
end

# compute_value_functions: Compute value functions for reference game
# For each reputation level, computes:
#   - V0: period 0 value (mix of types)
#   - V0_1: period 0 value for type 1
#   - V0_2: period 0 value for type 2
# Accounts for period 0 policies (mu0_1, mu0_2) and continuation values
# Returns named tuple (V, V1, V2)
function compute_value_functions(P0, mu0_1_grid, mu0_2_grid, mu1, mu2, P1, P2, params)
    (; mua_grid) = get_distribution(params)
    (; alpha, beta1, beta2, beta) = get_parameters(params)
    (; h, gamma1, gamma2) = get_inflation_costs(params)
    rho_grid = get_rho_grid(params)

    V_grid = similar(rho_grid)
    V1_grid = similar(rho_grid)
    V2_grid = similar(rho_grid)
   
    for rho_i in eachindex(rho_grid)
        rho = rho_grid[rho_i]
        
        HH_val1 = 0.0 
        HH_val2 = 0.0 

        HH_val0_1 = 0.0 
        HH_val0_2 = 0.0

        mu0_1 = mu0_1_grid[rho_i]
        mu0_2 = mu0_2_grid[rho_i]

        for mua_i in eachindex(mua_grid)
            mua = mua_grid[mua_i]

            cons1 = (1 + mua) / P1
            cons2 = (1 + mua) / P2

            cons0 = (1 + mua) / P0[rho_i]

            HH_val1 += (log(cons1) - alpha * cons1) * f(mua_i, mu1, params) 
            HH_val2 += (log(cons2) - alpha * cons2) * f(mua_i, mu2, params) 

            HH_val0_1 += (log(cons0) - alpha * cons0) * f(mua_i, mu0_1, params) 
            HH_val0_2 += (log(cons0) - alpha * cons0) * f(mua_i, mu0_2, params) 
        end

        EV1 = HH_val1 - gamma1 * h(mu1)
        EV2 = HH_val2 - gamma2 * h(mu2)

        EV = rho * HH_val1 + (1 - rho) * HH_val2

        V0_1 = (1 - beta1)  * (HH_val0_1 - gamma1 * h(mu0_1)) + beta1 * EV1
        V0_2 = (1 - beta2)  * (HH_val0_2 - gamma2 * h(mu0_2)) + beta2 * EV2

        V0 = (1 - beta) * (rho * HH_val0_1 + (1 - rho) * HH_val0_2) + beta * EV
        
        V_grid[rho_i] = V0
        V1_grid[rho_i] = V0_1
        V2_grid[rho_i] = V0_2
    end 

    return (; V = V_grid, V1 = V1_grid, V2 = V2_grid) 
end 

"""
    solve_reference_game(params)

Solve the reference game without reputation effects (full information benchmark).

Computes the equilibrium where types are known, serving as a comparison point
for the reputation game. Solves for period 1 steady state policies and prices,
then computes period 0 equilibrium for each reputation level.

# Arguments
- `params::ModelParameters`: Model parameters

# Returns
Named tuple with:
- `P0`: Period 0 price levels (vector over ρ grid)
- `mu0_1`, `mu0_2`: Period 0 policies for types 1 and 2
- `V`, `V1`, `V2`: Value functions
"""
function solve_reference_game(params)
    rho_grid = get_rho_grid(params)

    (; mu1, mu2, P1, P2) = solve_reference_period1(params)

    P0 = similar(rho_grid)
    mu0_1 = similar(rho_grid)
    mu0_2 = similar(rho_grid)

    for rho_i in eachindex(rho_grid)
        rho = rho_grid[rho_i]
        P0[rho_i] = find_zero(P -> period0_pricing_residual(P, mu1, mu2, rho, params), 1.0)
        mu0_1[rho_i], mu0_2[rho_i] = compute_mu0s(P0[rho_i], params)
    end

    values = compute_value_functions(P0, mu0_1, mu0_2, mu1, mu2, P1, P2, params)

    return (; P0, mu0_1, mu0_2, values...)
end


#####################################################################################
# Helper functions 


# Simple linear interpolation function
# Nothing more fancy is needed here
function linear_interp(x::AbstractVector, y::AbstractVector, xi)
    @assert length(x) == length(y) && length(x) ≥ 2
    i = searchsortedlast(x, xi)
    i == 0 && return y[1]                        # left clamp
    i ≥ length(x) && return y[end]               # right clamp
    t = (xi - x[i]) / (x[i+1] - x[i])
    return (1 - t) * y[i] + t * y[i+1]
end


# Helper function to compute distance between two policy objects
function get_distance(policies1, policies2)
    # Use a single thread-safe reduction computing max over all fields at each index
    return tmapreduce(max, eachindex(policies1.V1); init=0.0) do i
        max(
            abs(policies1.V1[i] - policies2.V1[i]),
            abs(policies1.V2[i] - policies2.V2[i]),
            abs(policies1.V[i] - policies2.V[i]),
            abs(policies1.P[i] - policies2.P[i]),
            abs(policies1.mu1_pol[i] - policies2.mu1_pol[i]),
            abs(policies1.mu2_pol[i] - policies2.mu2_pol[i])
        )
    end
end


"""
    ergodic_distribution(T; max_iters=10_000, tol=1e-10, init=ones(size(T,1))/size(T,1))

Compute the ergodic (stationary) distribution of a transition matrix `T`.

Uses power iteration to find the eigenvector corresponding to eigenvalue 1.
Warns if convergence is not achieved within `max_iters` iterations.

# Arguments
- `T`: Square transition matrix (rows sum to 1)
- `max_iters`: Maximum number of iterations
- `tol`: Convergence tolerance
- `init`: Initial distribution guess

# Returns
- Normalized stationary distribution vector
"""
function ergodic_distribution(T; max_iters = 10_000, tol = 1e-10, init = ones(size(T, 1))/size(T, 1))
    dist = Inf
    pi = init
    iter = 0

    for iter in 1:max_iters
        new_pi = T' * pi
        dist = maximum(abs.(new_pi .- pi))
        pi = new_pi
        (dist < tol) && break
    end
    (iter == max_iters && dist >= tol) && @warn "ergodic_distribution did not converge after $max_iters iterations (dist=$dist)"
    return pi / sum(pi)
end


# Helper function to compute expected inflation given current reputation index
# Used for checking inflation dynamics
function expected_inflation(rho_i, params, policies)
    (; mua_grid) = get_distribution(params)
    rho_grid = get_rho_grid(params)
    (; P) = policies

    exp_inflation = 0.0 
    for mua_i in eachindex(mua_grid)
        rho_prime = rhoplus(mua_i, rho_i, params, policies)
        P_next = (1 + mua_grid[mua_i]) * linear_interp(rho_grid, P, rho_prime)
        exp_inflation += (P_next / P[rho_i]) * f_given_policies(mua_i, rho_i, params, policies)
    end
    return exp_inflation - 1.0
end 


# Helper function to compute expected inflation density and average expected inflation
# Uses the ergodic distribution over reputation levels
function expected_inflation_density(params, policies, ergodic_rho)
    rho_grid = get_rho_grid(params)

    exp_pi = 100 .* [expected_inflation(rho_i, params, policies) for rho_i in eachindex(rho_grid)]

    ergodic_rho_density = ergodic_rho

    density = similar(exp_pi)
    for i in eachindex(exp_pi)
        if i == 1 
            density[i] = ergodic_rho_density[i] / ((exp_pi[i] - exp_pi[i+1]) / 2)
        elseif i == length(exp_pi)
            density[i] = ergodic_rho_density[i] / ((exp_pi[i-1] - exp_pi[i]) / 2)
        else
            density[i] = ergodic_rho_density[i] / ((exp_pi[i-1] - exp_pi[i+1]) / 2)
        end
    end
    average_exp_pi = sum(exp_pi .* ergodic_rho_density) 
    return exp_pi, density, average_exp_pi
end


# Helper function to compute the ergodic distribution of inflation
# Uses the ergodic distribution over reputation levels and the policies
function ergodic_inflation(params, policies; ergodic_rho = nothing, pimin = -0.02, pimax = 1.0, npoints = 2000, pi_grid = range(pimin, pimax; length = npoints) )

    rho_grid = get_rho_grid(params)
    mua_grid = get_distribution(params).mua_grid

    density = zeros(length(pi_grid))

    if ergodic_rho === nothing
        T, _, _ = build_transition_matrices(params, policies)
        ergodic_rho = ergodic_distribution(T)
    end

    average_pi = 0.0
    for rho_i in eachindex(rho_grid)
        for mua_i in eachindex(mua_grid)
            mua = mua_grid[mua_i]
            rho_prime = rhoplus(mua_i, rho_i, params, policies)
            P_next = linear_interp(rho_grid, policies.P, rho_prime)
            pi = (1 + mua) * (P_next / policies.P[rho_i]) - 1
       
            i = searchsortedlast(pi_grid, pi)
            i = clamp(i, 1, length(pi_grid))

            density[i] += ergodic_rho[rho_i] * f_given_policies(mua_i, rho_i, params, policies)
            average_pi += pi * ergodic_rho[rho_i] * f_given_policies(mua_i, rho_i, params, policies)
        end
    end
    
    density = density / sum(density)

    return (; grid = pi_grid, density = density, average = average_pi)
end