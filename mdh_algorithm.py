import numpy as np
from mpmath import mp, mpf, nstr, e, pi, j, exp, log, power, fabs, loggamma
import math

# --- Global precision ---
PRECISION = 300
mp.dps = PRECISION

# ====================================================
# 1. STABILIZED KAPPA FUNCTION (with mpmath.loggamma)
# ====================================================
kappa_cache = {}  # Cache for fast access

def kappa_stabilized(q):
    """Calculates kappa_q using mpmath.loggamma for stability and precision."""
    if q in kappa_cache:
        return kappa_cache[q]

    if q == 1:
        result = mpf(1)
    else:
        # Primitive q-th root of unity
        omega = e ** ((2 * pi * j) / q)

        # Sum of loggamma(3/2 - omega^j / 2) for j from 1 to q-1
        log_sum = mpf(0)
        for j_val in range(1, q):
            arg = mpf(3)/2 - (omega ** j_val) / 2
            # KEY: Use mpmath's loggamma which handles mp.mpc at full precision
            log_sum += loggamma(arg)

        # kappa_q = (2^q) / (pi^(q/2)) * exp(log_sum) - 1
        numerator = power(2, q)
        denominator = power(pi, q / 2)
        result = ((numerator / denominator) * exp(log_sum) - mpf(1)).real

    kappa_cache[q] = result
    return result

# ======================================================
# 2 MAIN GREEDY ALGORITHM with restarts
# ======================================================
def algorithm_restart_optimized(C, S0_initial, eps_max, max_restarts, max_q):
    """
    Main MHD algorithm function. Returns final approximation,
    list of corrections, and number of restarts.
    """
    C_scalar = mpf(C)
    S0 = mpf(S0_initial)
    total_corrections = []
    restart_count = 0

    # ================================================
    # SUB-FUNCTION: One Greedy pass
    # ================================================
    def greedy_pass(A_start, q_start):
        local_A = A_start
        local_q = q_start
        local_corrections = []

        while local_q <= max_q:
            local_improved = True
            while local_improved:
                local_improved = False
                kappa_val = kappa_stabilized(local_q)

                # Try addition
                if fabs(local_A + kappa_val - C_scalar) < fabs(local_A - C_scalar):
                    local_A += kappa_val
                    local_corrections.append((local_q, 1))
                    local_improved = True
                # Try subtraction
                elif fabs(local_A - kappa_val - C_scalar) < fabs(local_A - C_scalar):
                    local_A -= kappa_val
                    local_corrections.append((local_q, -1))
                    local_improved = True

            local_q += 1
            # Stopping check based on relative error
            if fabs((local_A - C_scalar) / C_scalar) <= eps_max:
                break

        return local_A, local_corrections

    # ================================================
    # HEADER PRINT
    # ================================================
    print("="*60)
    print("MHD ALGORITHM: mpmath LogGamma Stabilization")
    print("Target: C = ", nstr(C_scalar, 15))
    print("Initial S0 = ", nstr(S0, 15))
    print("Tolerance: epsilon_rel = ", eps_max)
    eps_abs = eps_max * fabs(C_scalar)
    print("           epsilon_abs = ", nstr(eps_abs, 3))
    print("Maximum q: ", max_q, ", Maximum restarts: ", max_restarts)
    print("="*60)

    # ================================================
    # MAIN LOOP WITH RESTARTS
    # ================================================
    while restart_count < max_restarts and fabs((S0 - C_scalar) / C_scalar) > eps_max:
        restart_count += 1

        print(f"\n*** RESTART {restart_count} ***")
        print(f"New S0 = ", nstr(S0, 10))
        print(f"Current abs. error: ", nstr(fabs(S0 - C_scalar), 5))
        print(f"Current rel. error: ", nstr(fabs((S0 - C_scalar) / C_scalar), 5))

        # Adaptive starting q
        curr_abs_err = fabs(S0 - C_scalar)
        q_start = 2
        while q_start < max_q and fabs(kappa_stabilized(q_start)) > curr_abs_err:
            q_start += 1
        print(f"  Starting from q = {q_start} (|kappa_q| ~~ {nstr(fabs(kappa_stabilized(q_start)), 3)})")

        # Execute greedy pass
        A, result_list = greedy_pass(S0, q_start)

        # Display restart results
        print(f"Restart {restart_count} completed.")
        print(f"Number of corrections in this restart: {len(result_list)}")
        total_corrections.extend(result_list)
        print(f"Cumulative corrections: {len(total_corrections)}")
        print(f"Current rel. error: {nstr(fabs((A - C_scalar) / C_scalar), 10)}")

        # Preparation for the next restart
        S0 = A

    # ================================================
    # FINAL RESULTS
    # ================================================
    final_A = A
    print("\n" + "="*60)
    print("FINAL RESULTS OF MHD ALGORITHM")
    print("="*60)
    print(f"Number of restarts: {restart_count}")
    print(f"Total corrections: {len(total_corrections)}")
    print(f"Final S0: {nstr(final_A, 40)}")
    print(f"Target C: {nstr(C_scalar, 40)}")

    # Errors
    error_abs = fabs(final_A - C_scalar)
    error_rel = error_abs / fabs(C_scalar)
    print("\nERRORS:")
    print(f"Absolute: Delta = {nstr(error_abs, 10)}")
    print(f"Relative: delta = {nstr(error_rel, 10)}")
    print(f"Number of accurate decimals: ~{int(-math.log10(float(error_abs)))}")

    # Hierarchical analysis
    if total_corrections:
        print("\nHIERARCHICAL STRUCTURE OF Y(4-methylheptane):")
        # Dominant descriptor
        dominant_q = total_corrections[0][0]
        m_star = sum(1 for q_val, coeff in total_corrections if q_val == dominant_q)
        dominant_sign = "+1" if total_corrections[0][1] == 1 else "-1"
        print(f"  Dominant descriptor: q* = {dominant_q}, m* = {m_star}, sign = {dominant_sign}")

        # Precise binary code
        binary_code = total_corrections
        pos_count = sum(1 for _, coeff in binary_code if coeff == 1)
        neg_count = sum(1 for _, coeff in binary_code if coeff == -1)
        print(f"  Precise binary code: {len(binary_code)} corrections")
        print(f"    (+1: {pos_count}, -1: {neg_count})")

        # Hierarchical list
        print("\nHIERARCHICAL LIST (q, c):")
        for i, (q_val, coeff) in enumerate(total_corrections, start=1):
            sign_str = "+1" if coeff == 1 else "-1"
            print(f"  {i}. ({q_val}, {sign_str})")

        # Export to file (optional)
        mhd_string = ", ".join([f"({q},{c})" for q, c in total_corrections])
        export_lines = [
            f"Y(4-methylheptane) MHD Code (Python V2 - mpmath):",
            mhd_string,
            "",
            f"Relative error: {nstr(error_rel, 5)}",
            f"Absolute error: {nstr(error_abs, 5)}",
            f"Number of corrections: {len(total_corrections)}",
            f"q range: {min(q for q, _ in total_corrections)} -> {max(q for q, _ in total_corrections)}"
        ]
        with open('4_methylheptane_MHD_code_mpmath.txt', 'w') as f:
            f.write("\n".join(export_lines))
        print("\n MHD code saved in '4_methylheptane_MHD_code_mpmath.txt'")

    return final_A, total_corrections, restart_count


# ====================================================
# 3. EXECUTION FOR 4-METILHEPTAN (Y = 1/S)
# ====================================================
if __name__ == "__main__":
    print("\n" + "#"*80)
    print("TEST 4-METILHEPTAN (Y = 1/S) - MHD ALGORITHM (Python with mpmath)")
    print("#"*80)

    # Parameters for 4-methylheptane
    S_4metilheptan = 109.32          # Standard molar entropy [cal/(mol K)]
    Y_target = mpf(1) / mpf(S_4metilheptan)   # Y = 1/S \in (0,1)
    S0_initial = mpf(0)              # Because Y < 1, floor is 0
    eps_target = 9.99e-17            # EXTREME precision target
    max_restarts = 30
    max_q = 120

    print("\nPARAMETERS:")
    print(f"  Entropija S(4-metilheptan) = {S_4metilheptan}")
    print(f"  Y = 1/S = {nstr(Y_target, 15)}")
    print(f"  S0initial = {S0_initial}")
    print(f"  epsilon_target = {eps_target}")
    print(f"  maxRestarts = {max_restarts}")
    print(f"  maxQ = {max_q}")

    # Execution
    print("\nStarting MHD ALGORITHM ...")
    result = algorithm_restart_optimized(
        Y_target, S0_initial, eps_target, max_restarts, max_q
    )

    # Verification
    if result and len(result) == 3:
        A_final, corrections, restarts = result
        print("\n" + "="*80)
        print("REPRODUCIBILITY VERIFICATION:")
        # Test 1: Coefficients are +-1
        all_binary = all(coeff in (1, -1) for _, coeff in corrections)
        print(f"   All coefficients are +-1: {all_binary}")
        # Test 2: Dominant descriptor
        if corrections:
            dominant_q = corrections[0][0]
            multiplicity = sum(1 for q_val, _ in corrections if q_val == dominant_q)
            print(f"   Dominant descriptor: q* = {dominant_q}, m* = {multiplicity}")
        else:
            print("   No corrections - empty hierarchical code")
        # Test 3: Accuracy
        final_error = fabs((A_final - Y_target) / Y_target)
        print(f"   Relative error: {nstr(final_error, 5)}")
        print(f"   Status: {'BELOW TARGET' if float(final_error) < eps_target else 'CLOSE TO TARGET'}")

    # Final MHD formula
    print("\n" + "=" * 80)
    print("FINAL MHD FORMULA FOR Y(4-metilheptan):")
    if corrections:
        # Grupisanje koeficijenata po q
        q_dict = {}
        for q_val, coeff in corrections:
            if q_val in q_dict:
                q_dict[q_val] += coeff
            else:
                q_dict[q_val] = coeff

        formula = "Y(4-metilheptan) ~~ S0 + "
        first_term = True

        for q_val in sorted(q_dict.keys()):
            total = q_dict[q_val]
            if total == 0:
                continue

            if total > 0:
                if first_term:
                    formula += f"{total}*kappa_{q_val}" if total != 1 else f"kappa_{q_val}"
                    first_term = False
                else:
                    formula += f" + {total}*kappa_{q_val}" if total != 1 else f" + kappa_{q_val}"
            else:  # total < 0
                abs_total = -total
                if first_term:
                    formula += f"-{abs_total}*kappa_{q_val}" if abs_total != 1 else f"-kappa_{q_val}"
                    first_term = False
                else:
                    formula += f" - {abs_total}*kappa_{q_val}" if abs_total != 1 else f" - kappa_{q_val}"

        formula += " + ..."
        print(f"  {formula}")
    else:
        print(f"  Y(4-metilheptan) ~~ S0 = {S0_initial}")