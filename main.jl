using Plots 
using Roots
using Base.Threads
using DelimitedFiles 


# Simple linear interpolation function
# Nothing more fancy is needed here
function lininterp(x::AbstractVector, y::AbstractVector, xi)
    @assert length(x) == length(y) && length(x) ≥ 2
    i = searchsortedlast(x, xi)
    i == 0 && return y[1]                         # left clamp
    i ≥ length(x) && return y[end]               # right clamp
    t = (xi - x[i]) / (x[i+1] - x[i])
    return (1 - t) * y[i] + t * y[i+1]
end


# Structs to hold parameters and policy/value functions

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
    hprime_inverse::F2 = (x) -> (x)

    d1::T = sum(lambda1 * exp(-lambda1 * mua) for mua in mua_grid)
    d2::T = sum(lambda2 * exp(-lambda2 * mua) for mua in mua_grid)
end

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



# The distribution function f and its derivative df
function f(mua_i, mu, params)
    (; mua_grid, lambda1, lambda2, d1, d2) = params
    mua = mua_grid[mua_i]
    return mu * lambda1 * exp(-lambda1 * mua) / d1 + (1 - mu) * lambda2 * exp(-lambda2 * mua) / d2
end

function df(mua_i, params)
    (; mua_grid, lambda1, lambda2, d1, d2) = params
    mua = mua_grid[mua_i]
    return lambda1 * exp(-lambda1 * mua) / d1 - lambda2 * exp(-lambda2 * mua) / d2
end

function f_given_policies(mua_i,  rho_i, params, policies)
    (; mu1_pol, mu2_pol) = policies
    (; rho_grid) = params
    rho = rho_grid[rho_i]
    return rho * f(mua_i, mu1_pol[rho_i], params) + (1 - rho) * f(mua_i, mu2_pol[rho_i], params)
end


###################################################################################################
# Solving the dynamic game


# Computing the g
function generate_g!(g, params, policies) 
    (; mua_grid) = params

    Threads.@threads for rho_i in eachindex(g)
        g[rho_i] = 0.0 
        for mua_i in eachindex(mua_grid)
            g[rho_i] += (1 / policies.c[mua_i, rho_i]) * f_given_policies(mua_i, rho_i, params, policies)
        end
    end 
end


# The belief updating functions rhohat and rhoplus
function rhohat(mua_i, rho_i, params, policies) 
    (; mu1_pol, mu2_pol) = policies
    rho = params.rho_grid[rho_i] 
    return rho * f(mua_i, mu1_pol[rho_i], params) / (
        rho * f(mua_i, mu1_pol[rho_i], params) + (1 - rho) * f(mua_i, mu2_pol[rho_i], params)
    )
end


function rhoplus(mua_i, rho_i, params, policies) 
    (; delta, epsilon) = params
    return (1 - delta) * rhohat(mua_i, rho_i, params, policies) + epsilon * (1 - rhohat(mua_i, rho_i, params, policies))
end


# Computing P times c (Pc) using the consumption optimality condition
function generate_Pc!(new_policies, policies, params) 
    (; rho_grid, mua_grid, beta) = params
    (; P, g) = policies
    (; Pc) = new_policies

    Threads.@threads for rho_i in eachindex(rho_grid)
        for mua_i in eachindex(mua_grid)
            mua = mua_grid[mua_i]
            Pc[mua_i, rho_i] = (1 + mua) * min(lininterp(rho_grid, P, rhoplus(mua_i, rho_i, params, policies)) / (beta * lininterp(rho_grid, g, rho_grid[rho_i])), 1)
        end
    end
end


# Computing P using the pricing optimality condition
function generate_P!(new_policies, policies, params) 
    (; mua_grid, rho_grid, beta, sigma, alpha) = params
    (; g, P) = policies
    (; Pc) = new_policies
    newP = new_policies.P

    Threads.@threads for j in eachindex(rho_grid)
        term1 = 0.0 
        term2 = 0.0 
        for i in eachindex(mua_grid)
            mua = mua_grid[i]
            rho_prime = rhoplus(i, j, params, policies)
            term1 += beta * (1- sigma) * (1/ (1 + mua)) * lininterp(rho_grid, g, rho_prime) / lininterp(rho_grid, P, rho_prime) * Pc[i, j] * f_given_policies(i, j, params, policies) 

            term2 += alpha * sigma * Pc[i, j] * f_given_policies(i, j, params, policies)
        end
        newP[j] = - term2 / term1
    end
end


# Constructing the new consumption function using Pc and P 
function generate_c!(new_policies, params)
    (; mua_grid, rho_grid) = params
    (; c, Pc, P) = new_policies

    Threads.@threads for j in eachindex(rho_grid)
        for i in eachindex(mua_grid)
            c[i, j] = Pc[i, j] / P[j]
        end 
    end 
end 


# Updating the value functions given fixed policies
# Used to construct the initial guess for the value functions
function update_values_fixed_policy!(new_policies, old_policies, params)
    (; mua_grid,  rho_grid, alpha, beta, beta1, beta2, gamma1, gamma2, h) = params
    (; V, V1, V2, mu1_pol, mu2_pol) = old_policies
    (; c) = new_policies
    
    Threads.@threads for rho_i in eachindex(rho_grid)
            mu1 = mu1_pol[rho_i]
            mu2 = mu2_pol[rho_i]
            value1 = 0.0
            value2 = 0.0
            value = 0.0
            for mua_i in eachindex(mua_grid) 
                rho_prime = rhoplus(mua_i, rho_i, params, old_policies)

                value += ((1 - beta) * (log(c[mua_i, rho_i]) - alpha * c[mua_i, rho_i]) + beta * lininterp(rho_grid, V, rho_prime)) * f_given_policies(mua_i, rho_i, params, old_policies)                

                value1 += ((1 - beta1) * (log(c[mua_i, rho_i]) - alpha * c[mua_i, rho_i]) + beta1 * lininterp(rho_grid, V1, rho_prime)) * f(mua_i, mu1, params)

                value2 += ((1 - beta2) * (log(c[mua_i, rho_i]) - alpha * c[mua_i, rho_i]) + beta2 * lininterp(rho_grid, V2, rho_prime)) * f(mua_i, mu2, params)
            end 
            value1 = value1 - (1- beta1) * gamma1 * h(mu1)  
            value2 = value2 - (1- beta2) * gamma2 * h(mu2)

        new_policies.mu1_pol[rho_i] = old_policies.mu1_pol[rho_i]
        new_policies.mu2_pol[rho_i] = old_policies.mu2_pol[rho_i]
        new_policies.V1[rho_i] = value1
        new_policies.V2[rho_i] = value2
        new_policies.V[rho_i] = value 
    end
end 


# Updating all equilibrium objects given old policies (but not computing optimal mu policy)
function update_all!(new_policies, old_policies, params)
    generate_g!(old_policies.g, params, old_policies)
    generate_Pc!(new_policies, old_policies, params)
    generate_P!(new_policies, old_policies, params)
    generate_c!(new_policies, params)
    
end 


# Computing optimal mu policy
function optimize_policies!(new_policies, old_policies, params)
    (; mua_grid, rho_grid, alpha, beta, beta1, beta2, gamma1, gamma2, h, hprime_inverse) = params
    (; V, V1, V2) = old_policies
    (; c) = new_policies
    
    Threads.@threads for rho_i in eachindex(rho_grid)
     
        # using the FOC to find optimal mu for each type 
        value1 = 0.0
        value2 = 0.0
        for mua_i in eachindex(mua_grid) 
            rho_prime = rhoplus(mua_i, rho_i, params, old_policies)

            value1 += ((1 - beta1) * (log(c[mua_i, rho_i]) - alpha * c[mua_i, rho_i]) + beta1 * lininterp(rho_grid, V1, rho_prime)) * df(mua_i, params)

            value2 += ((1 - beta2) * (log(c[mua_i, rho_i]) - alpha * c[mua_i, rho_i]) + beta2 * lininterp(rho_grid, V2, rho_prime)) * df(mua_i, params)
        end 
        value1 = value1 / ((1- beta1) * gamma1)   
        value2 = value2 / ((1- beta2) * gamma2) 
        
        best_mu1 = clamp(hprime_inverse(value1), 0.0, 1.0)
        best_mu2 = clamp(hprime_inverse(value2), 0.0, 1.0)

        new_policies.mu1_pol[rho_i] = best_mu1
        new_policies.mu2_pol[rho_i] = best_mu2

        # computing the value function at the optimal mu
        value1 = 0.0
        value2 = 0.0
        value = 0.0
        for mua_i in eachindex(mua_grid) 
            rho_prime = rhoplus(mua_i, rho_i, params, new_policies)

            value += ((1 - beta) * (log(c[mua_i, rho_i]) - alpha * c[mua_i, rho_i]) + beta * lininterp(rho_grid, V, rho_prime)) * f_given_policies(mua_i, rho_i, params, new_policies)

            value1 += ((1 - beta1) * (log(c[mua_i, rho_i]) - alpha * c[mua_i, rho_i]) + beta1 * lininterp(rho_grid, V1, rho_prime)) * f(mua_i, best_mu1, params)

            value2 += ((1 - beta2) * (log(c[mua_i, rho_i]) - alpha * c[mua_i, rho_i]) + beta2 * lininterp(rho_grid, V2, rho_prime)) * f(mua_i, best_mu2, params)
        end 
        best_value1 = value1 - ((1- beta1) * gamma1) * h(best_mu1)  
        best_value2 = value2 - ((1- beta2) * gamma2) * h(best_mu2)

        new_policies.V1[rho_i] = best_value1
        new_policies.V2[rho_i] = best_value2
        new_policies.V[rho_i] = value
    end
end 


# Iterating value function for fixed policy
function iterate_fixed_policy(params; max_iters=500, tol=1e-6, 
            old_policies = Policies(params), new_policies = Policies(params))

    println("Iterating value function for fixed policy...")
    for _ in 1:max_iters
        update_all!(new_policies, old_policies, params)
        update_values_fixed_policy!(new_policies, old_policies, params)

        diff = get_distance(new_policies, old_policies)
        # println("Iteration $iter, diff = $diff")
        (diff < tol) && break

        old_policies, new_policies = new_policies, old_policies
    end
    println("... done.")
    return new_policies, old_policies
end


# Iterating value and policy functions until convergence
# solving for optimal mu policies and with a gain parameter for convergence. 
function iterate_until_convergence(params; max_iters = 2_000, tol = 1e-6, 
            old_policies = Policies(params), new_policies = Policies(params), 
            gain = 0.5)

    @assert 0.0 < gain ≤ 1.0

    println("Iterating value and policy functions ...")

    for iter in 1:max_iters
        update_all!(new_policies, old_policies, params)
        optimize_policies!(new_policies, old_policies, params)

        diff = get_distance(new_policies, old_policies)
        println("Iteration $iter, diff = $diff")
        (diff < tol) && break

        if gain < 1.0
            # to facilitate convergence, do a convex combination of old and new values
            new_policies.V1 .= gain * new_policies.V1 .+ (1 - gain) * old_policies.V1
            new_policies.V2 .= gain * new_policies.V2 .+ (1 - gain) * old_policies.V2
            new_policies.P .= gain * new_policies.P .+ (1 - gain) * old_policies.P
        end

        old_policies, new_policies = new_policies, old_policies
    end

    println("... done.")
    return new_policies, old_policies
end


# Helper function to compute distance between two policy objects
function get_distance(policies1, policies2)
    dist = 0.0 
    Threads.@threads for i in eachindex(policies1.V1)
        dist = max(dist, abs(policies1.V1[i] - policies2.V1[i]))
        dist = max(dist, abs(policies1.V2[i] - policies2.V2[i]))
        dist = max(dist, abs(policies1.V[i] - policies2.V[i]))
        dist = max(dist, abs(policies1.P[i] - policies2.P[i]))
        dist = max(dist, abs(policies1.mu1_pol[i] - policies2.mu1_pol[i]))
        dist = max(dist, abs(policies1.mu2_pol[i] - policies2.mu2_pol[i]))
    end
    return dist
end


function create_transition_matrices(params, policies)
    (; mua_grid, rho_grid) = params
    (; mu1_pol, mu2_pol) = policies

    N = length(rho_grid)
    T = zeros(N, N)
    T1 = zeros(N, N)
    T2 = zeros(N, N)

    for rho_i in eachindex(rho_grid)
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
    end

    return T, T1, T2
end



function compute_ergodic(T; max_iters = 10_000, tol = 1e-10, init = ones(size(T, 1))/size(T, 1)) 
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


# Solving the dynamic game and computing the transition matrices
function solve_dynamic_game(params; policies_1 = Policies(params), policies_2 = Policies(params), max_iters = 2_000, tol = 1e-6, gain = 0.5)
    optimal_policies, _ = iterate_until_convergence(params; old_policies = policies_1, new_policies = policies_2, max_iters, tol, gain)
    T, T1, T2 = create_transition_matrices(params, optimal_policies)
    return optimal_policies, (; T, T1, T2)
end


# computing mean paths from an initial rho value 
function compute_rho_paths(rho_init, dynamics, params; periods = 100)
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


###################################################################################################
# Solving the reference game (without reputation effects)

function get_E1plusmua(params, mu)
    (; mua_grid) = params
    val = 0.0 
    for Emua_i in eachindex(mua_grid)
        Emua = mua_grid[Emua_i]
        val += (1 + Emua) * f(Emua_i, mu, params)
    end
    return val 
end

function get_E1over1plusmua(params, mu)
    (; mua_grid) = params
    val = 0.0 
    for Emua_i in eachindex(mua_grid)
        Emua = mua_grid[Emua_i]
        val += 1/(1 + Emua) * f(Emua_i, mu, params)
    end
    return val 
end

function part_of_c(params, mu)
    (; alpha, sigma, beta, mua_grid) = params
    val1 = 0.0 
    val2 = 0.0 
    for Emua_i in eachindex(mua_grid)
        Emua = mua_grid[Emua_i]
        val1 += 1/(1 + Emua) * f(Emua_i, mu, params) 
        val2 += (1 + Emua) * f(Emua_i, mu, params)
    end
    return 1/alpha * (sigma - 1)/sigma * beta * get_E1over1plusmua(params, mu) / get_E1plusmua(params, mu) 
end 

function optimality_mu(mu, params, gamma)
    value = 0.0 
    part = part_of_c(params, mu)
    for mua_i in eachindex(params.mua_grid)
        mua = params.mua_grid[mua_i]
        cons = part * (1 + mua)
        value += (log(cons) - params.alpha * cons) * df(mua_i, params)
    end
    value = value -  gamma * mu
    return value 
end 

function solve_period_1(params)    
    mu1 = find_zero(x -> optimality_mu(x, params, params.gamma1), 0.5)
    mu2 = find_zero(x -> optimality_mu(x, params, params.gamma2), 0.5)
    P1 = 1 / part_of_c(params, mu1)
    P2 = 1 / part_of_c(params, mu2)
    return (; mu1, mu2, P1, P2) 
end 

function get_mu0s(P, params)
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

function root_for_P(P, mu1, mu2, rho, params)
    (; alpha, sigma, beta, mua_grid) = params
    mu0_1, mu0_2 = get_mu0s(P, params)
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
    return alpha * sigma / (sigma - 1) * 1 / beta *  (rho * num_term1 + (1 - rho) * num_term2) / (rho * den_term1 + (1 - rho) * den_term2) - P 
end

function computing_value_funcions(P0, mu0_1_grid, mu0_2_grid, mu1, mu2, P1, P2, params)
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

function solve_reference_game(params; mu1 = nothing, mu2 = nothing)
    (; rho_grid) = params

    if mu1 === nothing || mu2 === nothing
        (; mu1, mu2, P1, P2) = solve_period_1(params)
    end 

    P0 = similar(rho_grid)
    mu0_1 = similar(rho_grid)
    mu0_2 = similar(rho_grid)

    for rho_i in eachindex(rho_grid)
        rho = rho_grid[rho_i]
        P0[rho_i] = find_zero(P -> root_for_P(P, mu1, mu2, rho, params), 1.0)
        mu0_1[rho_i], mu0_2[rho_i] = get_mu0s(P0[rho_i], params)
    end

    values = computing_value_funcions(P0, mu0_1, mu0_2, mu1, mu2, P1, P2, params)

    return (; P0, mu0_1, mu0_2, values...)
end



###################################################################################################
# Main simulation


params = Parameters()

@time optimal_policies, dynamics = solve_dynamic_game(params)
reference = solve_reference_game(params)

ergodic = compute_ergodic(dynamics.T)  # unconditional
ergodic1 = compute_ergodic(dynamics.T1)  # type 1
ergodic2 = compute_ergodic(dynamics.T2)  # type 2


# Plots 

# P
f1 = plot(params.rho_grid, optimal_policies.P, xlabel="rho", ylabel="Price Level P", linewidth = 2, label = "")
plot!(f1, params.rho_grid, reference.P0, xlabel="rho", ylabel="Price Level P", title="Reference Price Level vs. Reputation rho", linewidth = 2, label = "", linestyle = :dash)


# mu
f2 = plot(params.rho_grid, optimal_policies.mu1_pol, xlabel="rho", ylabel="mu", title="Optimal mu1 vs. Reputation rho", linewidth = 2, label = "" )
plot!(f2, params.rho_grid, optimal_policies.mu2_pol, linewidth = 2, label = "")
plot!(f2, params.rho_grid, reference.mu0_1, label = "", linestyle = :dash, linewidth = 2)
plot!(f2, params.rho_grid, reference.mu0_2, label = "", linestyle = :dash, linewidth = 2)


# g 
f3 = plot(params.rho_grid, optimal_policies.g, xlabel="rho", ylabel="g", title="g Function vs. Reputation rho", linewidth = 2)


# V1
f4a = plot(params.rho_grid, optimal_policies.V1, xlabel="rho", ylabel="V", title="Value Function V1, V2 vs. Reputation rho", linewidth = 2, label = "V1")
plot!(f4a, params.rho_grid, reference.V1, linewidth = 2, label = "Ref V1", linestyle = :dash)


# V2
f4b = plot(params.rho_grid, optimal_policies.V2, xlabel="rho", title="Value Function V2 vs. Reputation rho", linewidth = 2, label = "V2")
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
paths1 = compute_rho_paths(0.5, dynamics, params; periods = 100)
rho_path, rho_path_1, rho_path_2 = paths1
f6 = plot(rho_path, xlabel="Time Periods", ylabel="Reputation rho", title="Reputation Path over Time", linewidth = 2, label = "Overall")
plot!(f6, rho_path_1, linewidth = 2, label = "Type 1", linestyle = :dash)
plot!(f6, rho_path_2, linewidth = 2, label = "Type 2", linestyle = :dashdot)


# Reputation paths from initial value 0.2
paths2 = compute_rho_paths(0.2, dynamics, params; periods = 100)
rho_path, rho_path_1, rho_path_2 = paths2
f7 = plot(rho_path, xlabel="Time Periods", ylabel="Reputation rho", title="Reputation Path over Time", linewidth = 2, label = "Overall")
plot!(f7, rho_path_1, linewidth = 2, label = "Type 1", linestyle = :dash)
plot!(f7, rho_path_2, linewidth = 2, label = "Type 2", linestyle = :dashdot)


for f in [f1, f2, f3, f4a, f4b, f4c, f5, f6, f7]
    display(f)
end


# Exporting data to CSV files

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

