using Roots: find_zero
using OhMyThreads: @tasks, tmapreduce


# Structs to hold parameters and policy/value functions

# Parameters struct: holds all model parameters and grids
Base.@kwdef struct Parameters{T, V2, V3, F, F2}
    epsilon::T = 0.02
    delta::T = 0.02
    sigma::T = 5.0
    beta::T = 0.99
    gamma1::T = 0.0173
    gamma2::T = 0.0093 
    alpha::T = 1.0
    beta1::T = beta * (1 - delta)
    beta2::T = beta * (1 - epsilon)

    lambda1::T = 1 / 0.04
    lambda2::T = 1 / 0.01

    rho_grid::V2 = range(epsilon, stop = 1 - delta, length = 200)
    mua_grid::V3 = range(0.0, stop = 0.5, length = 2000)

    h::F = (mu) -> 1/2 * mu^2
    inv_hprime::F2 = (x) -> (x)

    # normalizing constants for the exponential distributions
    d1::T = sum(lambda1 * exp(-lambda1 * mua) for mua in mua_grid)
    d2::T = sum(lambda2 * exp(-lambda2 * mua) for mua in mua_grid)

    # Precompute density components to avoid repeated exp() in hot loops
    comp1_pdf::Vector{T} = [lambda1 * exp(-lambda1 * mua) / d1 for mua in mua_grid]
    comp2_pdf::Vector{T} = [lambda2 * exp(-lambda2 * mua) / d2 for mua in mua_grid]
end

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
Policies(params) = Policies(
    [0.0 for _ in params.rho_grid],
    [0.0 for _ in params.rho_grid],
    [1.0 for _ in params.rho_grid],
    [- params.alpha for _ in params.rho_grid],
    [- params.alpha for _ in params.rho_grid],
    [- params.alpha for _ in params.rho_grid],
    ones(length(params.mua_grid), length(params.rho_grid)), 
    ones(length(params.mua_grid), length(params.rho_grid)),  
    [1.0 for _ in params.rho_grid]
)


#####################################################################################
# Distribution functions

# f: Distribution function evaluated at grid index mua_i with mixture parameter mu
# Returns weighted average of two exponential distributions
f(mua_i, mu, params) = mu * params.comp1_pdf[mua_i] + (1 - mu) * params.comp2_pdf[mua_i]

# df: Derivative of distribution function with respect to mu
# Used in FOC for optimal policy
df(mua_i, params) = params.comp1_pdf[mua_i] - params.comp2_pdf[mua_i]

# f_given_policies: Distribution function given current policy functions
# Computes the implied distribution when types follow their optimal policies
function f_given_policies(mua_i,  rho_i, params, policies)
    (; mu1_pol, mu2_pol) = policies
    (; rho_grid) = params
    rho = rho_grid[rho_i]
    return rho * f(mua_i, mu1_pol[rho_i], params) + (1 - rho) * f(mua_i, mu2_pol[rho_i], params)
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
    rho = params.rho_grid[rho_i] 
    return rho * f(mua_i, mu1_pol[rho_i], params) / (
        rho * f(mua_i, mu1_pol[rho_i], params) + (1 - rho) * f(mua_i, mu2_pol[rho_i], params)
    )
end


# rhoplus: Updated belief accounting for type transitions
# Incorporates probability delta that type 1 becomes type 2
# and probability epsilon that type 2 becomes type 1
function rhoplus(mua_i, rho_i, params, policies) 
    (; delta, epsilon) = params
    return (1 - delta) * rhohat(mua_i, rho_i, params, policies) + epsilon * (1 - rhohat(mua_i, rho_i, params, policies))
end


#####################################################################################
# Equilibrium object computation

# update_g!: Compute the g function (expectation of 1/c)
# Used in the consumption optimality condition
# Modifies g in-place
function update_g!(g, params, policies) 
    (; mua_grid) = params
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
    (; rho_grid, mua_grid, beta) = params
    (; P, g) = policies
    (; Pc) = new_policies

    @tasks for rho_i in eachindex(rho_grid)
        for mua_i in eachindex(mua_grid)
            mua = mua_grid[mua_i]
            Pc[mua_i, rho_i] = (1 + mua) * min(linear_interp(rho_grid, P, rhoplus(mua_i, rho_i, params, policies)) / (beta * linear_interp(rho_grid, g, rho_grid[rho_i])), 1)
        end
    end
end


# update_P!: Compute price level P using the pricing optimality condition
# Firms choose prices optimally given demand and expectations
# Modifies new_policies.P in-place
function update_P!(new_policies, policies, params) 
    (; mua_grid, rho_grid, beta, sigma, alpha) = params
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
    (; mua_grid, rho_grid) = params
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
    (; mua_grid, rho_grid, alpha, beta, beta1, beta2, gamma1, gamma2, h, inv_hprime) = params
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
            gain = 0.5, verbose = true)

    @assert 0.0 < gain ≤ 1.0
    verbose && @info "Iterating value and policy functions ..."

    iter = 1
    diff = 0.0
    while iter <= max_iters
        update_all!(new_policies, old_policies, params)
        optimize_policies!(new_policies, old_policies, params)

        diff = get_distance(new_policies, old_policies)
        verbose && (iter % 10 == 0) && @info("    Iteration $iter, diff = $diff")
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
    iter == max_iters && @warn("Maximum iterations reached without convergence.")
    verbose && @info "... done."
    return new_policies, old_policies, diff
end


#####################################################################################
# Dynamics and transition matrices

# build_transition_matrices: Build transition matrices for reputation dynamics
# For each current reputation level and money shock:
#   - Compute next period's reputation (rhoplus)
#   - Distribute probability mass using linear interpolation
#   - Create separate matrices for overall (T), type 1 (T1), and type 2 (T2)
# Rows are normalized to sum to 1 (valid probability distribution)
# Returns: T, T1, T2 (N×N matrices where N = length(rho_grid))
function build_transition_matrices(params, policies)
    (; mua_grid, rho_grid) = params
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


# solve_dynamic_game: Main entry point for solving the dynamic game
# Calls iterate_until_convergence to find optimal policies
# Then computes transition matrices for reputation dynamics
# Returns: optimal_policies, named tuple with transition matrices (T, T1, T2)
function solve_dynamic_game(params; policies_1 = Policies(params), policies_2 = Policies(params), max_iters = 5_000, tol = 1e-6, gain = 0.5, verbose = true)
    optimal_policies, _, diff = iterate_until_convergence(params; old_policies = policies_1, new_policies = policies_2, max_iters, tol, gain, verbose)
    T, T1, T2 = build_transition_matrices(params, optimal_policies)
    return optimal_policies, (; T, T1, T2), diff
end


# simulate_rho_paths: Simulate mean reputation paths over time
# Starting from initial reputation rho_init, evolves the distribution forward
# Computes three paths:
#   - Overall path (mixing both types according to rho)
#   - Type 1 conditional path
#   - Type 2 conditional path
# Returns: (rho_path, rho_path_1, rho_path_2) - vectors of length periods+1
function simulate_rho_paths(rho_init, dynamics, params; periods = 100)
    (; rho_grid) = params

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

    for iter in 1:periods
        rho_vector = dynamics.T' * rho_vector
        rho_vector_1 = dynamics.T1' * rho_vector_1
        rho_vector_2 = dynamics.T2' * rho_vector_2

        rho_next = sum(rho_grid .* rho_vector)
        rho_next_1 = sum(rho_grid .* rho_vector_1)
        rho_next_2 = sum(rho_grid .* rho_vector_2)

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
    (; mua_grid) = params
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
    (; mua_grid) = params
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
    (; alpha, sigma, beta, mua_grid) = params
    val1 = 0.0 
    val2 = 0.0 
    for Emua_i in eachindex(mua_grid)
        Emua = mua_grid[Emua_i]
        val1 += 1/(1 + Emua) * f(Emua_i, mu, params) 
        val2 += (1 + Emua) * f(Emua_i, mu, params)
    end
    return 1/alpha * (sigma - 1)/sigma * beta * expected_one_over_one_plus_mua(params, mu) / expected_one_plus_mua(params, mu) 
end 

# mu_foc: FOC for optimal mu in reference game (period 1)
# Returns derivative of objective with respect to mu
# Used with root finder to find optimal policy
function mu_foc(mu, params, gamma)
    value = 0.0 
    part = compute_consumption_scaler(params, mu)
    for mua_i in eachindex(params.mua_grid)
        mua = params.mua_grid[mua_i]
        cons = part * (1 + mua)
        value += (log(cons) - params.alpha * cons) * df(mua_i, params)
    end
    value = value -  gamma * mu
    return value 
end 

# solve_reference_period1: Solve for period 1 equilibrium in reference game
# Finds optimal mu1, mu2 and corresponding prices P1, P2
# Uses root finding on FOC for each type
# Returns named tuple (mu1, mu2, P1, P2)
function solve_reference_period1(params)    
    mu1 = find_zero(x -> mu_foc(x, params, params.gamma1), 0.5)
    mu2 = find_zero(x -> mu_foc(x, params, params.gamma2), 0.5)
    P1 = 1 / compute_consumption_scaler(params, mu1)
    P2 = 1 / compute_consumption_scaler(params, mu2)
    return (; mu1, mu2, P1, P2) 
end 

# compute_mu0s: Compute period 0 optimal policies given price P
# Solves for mu that satisfies FOC given P
# Returns mu01 (type 1) and mu02 (type 2)
function compute_mu0s(P, params)
    (; alpha, mua_grid, gamma1, gamma2) = params
    val = 0.0 
    for mua_i in eachindex(mua_grid)
        mua = mua_grid[mua_i] 
        val += (log(1 + mua) - alpha * (1 + mua) / P) * df(mua_i, params)
    end
    mu01 = val / gamma1
    mu02 = val / gamma2 
    return mu01, mu02
end 

# period0_pricing_residual: Pricing equation for period 0 in reference game
# Given mu1, mu2 (period 1 policies), rho (reputation):
#   Given P0 obtain mu0_1 and mu0_2
#   Given mu01 and mu02,  computes equilibrium P0' from pricing FOC
#   Returns residual P0' - P0 (should be zero at equilibrium)
function period0_pricing_residual(P0, mu1, mu2, rho, params)
    (; alpha, sigma, beta, mua_grid) = params
    mu0_1, mu0_2 = compute_mu0s(P0, params)
    num_term1 = 0.0 
    num_term2 = 0.0
    den_term1 = 0.0 
    den_term2 = 0.0 

    for mua_i in eachindex(mua_grid)
        mua = params.mua_grid[mua_i]
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
    (; mua_grid, alpha, beta1, beta2, beta, h, gamma1, gamma2) = params

    V_grid = similar(params.rho_grid)
    V1_grid = similar(params.rho_grid)
    V2_grid = similar(params.rho_grid)
   
    for rho_i in eachindex(params.rho_grid)
        rho = params.rho_grid[rho_i]
        
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

# solve_reference_game: Main solver for reference game without reputation effects
# If mu1 and mu2 are not provided, solves for period 1 equilibrium first
function solve_reference_game(params; mu1 = nothing, mu2 = nothing)
    (; rho_grid) = params

    if mu1 === nothing || mu2 === nothing
        (; mu1, mu2, P1, P2) = solve_reference_period1(params)
    end 

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


# Helper function to compute the ergodic distribution given a transition matrix
function ergodic_distribution(T; max_iters = 10_000, tol = 1e-10, init = ones(size(T, 1))/size(T, 1)) 
    dist = Inf
    pi = init

    for iter in 1:max_iters
        new_pi = T' * pi
        dist = maximum(abs.(new_pi .- pi))
        pi = new_pi
        (dist < tol) && break
    end
    return pi / sum(pi)
end

