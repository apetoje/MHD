import math
from mpmath import mp, mpf, nstr, e, pi, j, exp, power, fabs, loggamma

# ----------------------------------------------------------------------
# Global precision
PRECISION = 300
mp.dps = PRECISION

# Cache for kappa values
kappa_cache = {}

def kappa(q):
    """
    Returns the basis element κ_q defined by
      κ_1 = 1,
      κ_q = \overline{\mathcal{K}}_q - 1   for q >= 2,
    using the closed-form expression from Srivastava & Petojević (2026).
    """
    if q in kappa_cache:
        return kappa_cache[q]

    if q == 1:
        result = mpf(1)
    else:
        # Primitive q-th root of unity
        omega = e ** ((2 * pi * j) / q)
        # Sum of logΓ(3/2 - ω^j / 2) for j = 1, …, q-1
        log_sum = mpf(0)
        for j_val in range(1, q):
            arg = mpf(3)/2 - (omega ** j_val) / 2
            log_sum += loggamma(arg)
        # κ_q = (2^q / π^{q/2}) * exp(log_sum) - 1
        numerator = power(2, q)
        denominator = power(pi, q / 2)
        result = ((numerator / denominator) * exp(log_sum) - mpf(1)).real

    kappa_cache[q] = result
    return result


def mhd_decomposition(C, epsilon, Q_max, max_restarts=10, adaptive_start=True):
    """
    Hierarchical Decomposition Method (MHD) of a positive constant C
    using the basis {κ_q}.

    Parameters
    ----------
    C : float or mpf
        Target constant to approximate.
    epsilon : float
        Desired relative accuracy.
    Q_max : int
        Maximum index q of the basis.
    max_restarts : int
        Maximum number of restarts.
    adaptive_start : bool
        If True, start from the first q where |κ_q| is comparable to the
        current absolute error.

    Returns
    -------
    A_final : mpf
        Final approximation.
    m_sequence : list of (q, m_q)
        Coefficients m_q for q ≥ 2 with m_q ≠ 0, sorted by q.
    total_restarts : int
        Number of restarts performed.
    """
    C = mpf(C)
    # Initial approximation – integer part (κ_1 = 1, m_1 = ⌊C⌋)
    m_1 = int(math.floor(C))
    A = mpf(m_1)

    # Accumulated MHD coefficients
    m_q_list = []          # will store (q, sign) from each pass
    total_restarts = 0
    rel_error = fabs((A - C) / C)

    while total_restarts < max_restarts and rel_error > epsilon:
        # --- Adaptive start (only for the first pass) ---
        if adaptive_start and total_restarts == 0:
            curr_abs_err = fabs(A - C)
            q_start = 2
            while q_start < Q_max and fabs(kappa(q_start)) > curr_abs_err:
                q_start += 1
        else:
            q_start = 2

        # --- One greedy pass ---
        A_pass = A
        pass_corrections = []
        q = q_start
        while q <= Q_max:
            improved = True
            while improved:
                improved = False
                kq = kappa(q)
                # Try to add
                if fabs(A_pass + kq - C) < fabs(A_pass - C):
                    A_pass += kq
                    pass_corrections.append((q, +1))
                    improved = True
                # Try to subtract
                elif fabs(A_pass - kq - C) < fabs(A_pass - C):
                    A_pass -= kq
                    pass_corrections.append((q, -1))
                    improved = True
            q += 1
            # Early stop if accuracy reached
            if fabs((A_pass - C) / C) <= epsilon:
                break

        # Absorb this pass into the global list
        m_q_list.extend(pass_corrections)
        A = A_pass
        total_restarts += 1
        rel_error = fabs((A - C) / C)

    # Aggregate coefficients: sum signs for each q
    m_dict = {}
    for q, sgn in m_q_list:
        m_dict[q] = m_dict.get(q, 0) + sgn

    # Build sorted list of (q, m_q) with non-zero m_q
    m_sequence = [(q, m_dict[q]) for q in sorted(m_dict.keys()) if m_dict[q] != 0]

    # Final approximation computed from the formula
    A_final = m_1
    for q, m in m_sequence:
        A_final += m * kappa(q)

    return A_final, m_sequence, total_restarts


# ======================================================================
# Example: 4‑methylheptane
# ======================================================================
if __name__ == "__main__":
    # Standard molar entropy of 4-methylheptane (Scott's correlation)
    S_4metilheptan = 109.32          # cal/(mol·K)
    Y_target = mpf(1) / mpf(S_4metilheptan)   # C = 1/S ∈ (0,1)

    eps_target = 9.99e-17
    Q_max = 120
    max_restarts = 30

    print("="*70)
    print("MHD DECOMPOSITION – 4‑methylheptane")
    print("="*70)
    print(f"S(4‑methylheptane) = {S_4metilheptan} cal mol⁻¹ K⁻¹")
    print(f"Target constant C = 1/S = {nstr(Y_target, 20)}")
    print(f"Tolerance ε = {eps_target}")
    print(f"Q_max = {Q_max}")
    print()

    A_final, m_seq, restarts = mhd_decomposition(
        Y_target, eps_target, Q_max, max_restarts, adaptive_start=True
    )

    # Results
    print(f"Restarts performed: {restarts}")
    print(f"Number of non‑zero m_q (q ≥ 2): {len(m_seq)}")
    print(f"Final approximation A = {nstr(A_final, 45)}")
    print(f"Target C             = {nstr(Y_target, 45)}")

    error_abs = fabs(A_final - Y_target)
    error_rel = error_abs / fabs(Y_target)
    print(f"\nAbsolute error: {nstr(error_abs, 10)}")
    print(f"Relative error: {nstr(error_rel, 10)}")
    print(f"≈ {int(-math.log10(float(error_abs)))} correct decimal places")

    # MHD coefficients
    print("\nMHD code (q, m_q) for q ≥ 2:")
    for q, m in m_seq:
        print(f"  q = {q:3d}    m_{q} = {m:+d}")
    print()

    # Formula in compact form
    terms = []
    for q, m in m_seq:
        if m == 1:
            terms.append(f"κ_{q}")
        elif m == -1:
            terms.append(f"-κ_{q}")
        else:
            terms.append(f"{m:+d}·κ_{q}")
    formula = "C ≈ " + " + ".join(terms).replace("+ -", "- ")
    print("MHD formula (truncated):")
    print(formula)
    print("… with further corrections beyond shown terms.")
