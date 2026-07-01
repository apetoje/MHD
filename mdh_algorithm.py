import math
from mpmath import mp, mpf, nstr, e, pi, j, exp, power, fabs, loggamma

# ----------------------------------------------------------------------
# Global precision
PRECISION = 300
mp.dps = PRECISION

# Cache for kappa values
kappa_cache = {}

def kappa(q):
    r"""
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
        omega = e ** ((2 * pi * j) / q)
        log_sum = mpf(0)
        for j_val in range(1, q):
            arg = mpf(3)/2 - (omega ** j_val) / 2
            log_sum += loggamma(arg)
        numerator = power(2, q)
        denominator = power(pi, q / 2)
        result = ((numerator / denominator) * exp(log_sum) - mpf(1)).real

    kappa_cache[q] = result
    return result


def mhd_single_pass(C, epsilon, Q_max):
    """
    MHD single-pass decomposition – exactly follows the basic pseudocode.
    """
    C = mpf(C)
    m = [0] * (Q_max + 1)
    m[1] = int(math.floor(C))
    A = mpf(m[1])

    for q in range(2, Q_max + 1):
        improved = True
        while improved:
            improved = False
            kq = kappa(q)
            if fabs(A + kq - C) < fabs(A - C):
                A += kq
                m[q] += 1
                improved = True
            elif fabs(A - kq - C) < fabs(A - C):
                A -= kq
                m[q] -= 1
                improved = True
        if fabs(A - C) / fabs(C) < epsilon:
            break

    return m, A


# ======================================================================
# Example: 4‑methylheptane
# ======================================================================
if __name__ == "__main__":
    S_4metilheptan = 109.32          # cal/(mol·K)
    C_target = mpf(1) / mpf(S_4metilheptan)

    eps_target = 9.99e-17
    Q_max = 120

    print("=" * 70)
    print("MHD SINGLE-PASS DECOMPOSITION – 4‑methylheptane")
    print("=" * 70)
    print(f"S(4‑methylheptane) = {S_4metilheptan} cal mol⁻¹ K⁻¹")
    print(f"Target constant C = 1/S = {nstr(C_target, 20)}")
    print(f"Tolerance ε = {eps_target}")
    print(f"Q_max = {Q_max}")
    print()

    m_list, A_final = mhd_single_pass(C_target, eps_target, Q_max)

    # Find the last index with non‑zero coefficient (or at least 1)
    last_q = max((q for q in range(1, Q_max+1) if m_list[q] != 0), default=1)

    # Print integer sequence from m_1 to m_last
    int_seq = ", ".join(str(m_list[q]) for q in range(1, last_q+1))
    print(f"MHD integer sequence (m₁ … m_{last_q}):")
    print(f"  ({int_seq})")
    print()

    # Non‑zero coefficients for q ≥ 2
    m_seq = [(q, m_list[q]) for q in range(2, last_q+1) if m_list[q] != 0]

    print(f"m_1 = {m_list[1]}")
    print(f"Number of non‑zero m_q (q ≥ 2): {len(m_seq)}")
    print(f"Final approximation A = {nstr(A_final, 45)}")
    print(f"Target C             = {nstr(C_target, 45)}")

    error_abs = fabs(A_final - C_target)
    error_rel = error_abs / fabs(C_target)
    print(f"\nAbsolute error: {nstr(error_abs, 10)}")
    print(f"Relative error: {nstr(error_rel, 10)}")
    print(f"≈ {int(-math.log10(float(error_abs)))} correct decimal places")

    print("\nMHD code (q, m_q) for q ≥ 2:")
    for q, m_val in m_seq:
        print(f"  q = {q:3d}    m_{q} = {m_val:+d}")
    print()

    # Compact formula
    terms = []
    for q, m_val in m_seq:
        if m_val == 1:
            terms.append(f"κ_{q}")
        elif m_val == -1:
            terms.append(f"-κ_{q}")
        elif m_val > 0:
            terms.append(f"+{m_val}·κ_{q}")
        else:
            terms.append(f"{m_val}·κ_{q}")
    formula = "C ≈ " + " ".join(terms).strip().replace(" - ", " - ")
    print("MHD formula (truncated):")
    print(formula)
    print("… with further corrections beyond shown terms.")
