#=
Test Suite for Monetary Reputation Model

This file tests the model behavior in a special case where both types have identical
cost parameters (gamma1 = gamma2). In this symmetric case, the dynamic game solution
should collapse to the reference game (no-reputation) solution.

Key theoretical prediction:
When gamma1 = gamma2, reputation becomes uninformative because both types behave 
identically. Therefore:
  - Optimal policies should match reference policies: mu1 = mu2 = mu0_1 = mu0_2
  - Prices should match: P = P0 
  - Value functions should match: V1 = V2 = V1_ref = V2_ref
  - Household value function should match: V = V_ref

This serves as a consistency check that the model correctly handles the degenerate
case where reputation effects disappear.
=#

# Test Case: Symmetric types (gamma1 = gamma2 = 0.02)
params1 = Parameters(gamma1 = 0.02, gamma2 = 0.02)

# Solve the dynamic game with reputation effects
@time optimal_policies1, dynamics1 = solve_dynamic_game(params1, tol = 1e-7)

# Solve the reference game (no reputation effects)
reference = solve_reference_game(params1)

# Tolerance for numerical comparisons
tol = 1e-5 

# Test 1: Price levels should match
# When types are identical, the price policy in the dynamic game (P) should equal
# the reference game price (P0) across all reputation levels
@assert maximum(abs.(optimal_policies1.P  .- reference.P0)) < tol  

# Test 2: Type 1 policy should match reference policy
# Optimal money injection for type 1 (mu1_pol) should match the reference (mu0_1)
@assert maximum(abs.(optimal_policies1.mu1_pol  .- reference.mu0_1)) < tol

# Test 3: Type 2 policy should match reference policy
# Optimal money injection for type 2 (mu2_pol) should match the reference (mu0_2)
@assert maximum(abs.(optimal_policies1.mu2_pol  .- reference.mu0_2)) < tol

# Test 4: Both types should behave identically
# Since gamma1 = gamma2, both types should choose the same policy at every reputation level
@assert maximum(abs.(optimal_policies1.mu2_pol  .- optimal_policies1.mu1_pol)) < tol

# Test 5: Type 1 value function should match reference
# Value function for type 1 (V1) should equal the reference value (V1_ref)
@assert maximum(abs.(optimal_policies1.V1  .- reference.V1)) < tol

# Test 6: Type 2 value function should match reference
# Value function for type 2 (V2) should equal the reference value (V2_ref)
@assert maximum(abs.(optimal_policies1.V2  .- reference.V2)) < tol

# Test 7: Household value function should match reference
# Household value function (V) should equal the reference value (V_ref)
@assert maximum(abs.(optimal_policies1.V  .- reference.V)) < tol

println("All tests passed for gamma1 = gamma2 = 0.02")
