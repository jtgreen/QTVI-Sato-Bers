"""
Sato-Bers ventricular cardiac myocyte model — Python implementation.

Direct port of the C++ implementation (cell.h / cell.cc) preserving identical
numerics, variable names, and algorithm structure.

Model overview:
    This module implements a single ventricular cardiac myocyte based on the
    Sato-Bers model. The cell has 15 state variables tracking membrane
    potential, intracellular calcium in four compartments, SR release current,
    and gating variables for seven ion currents.

Ion currents modeled:
    - INa  : Fast Na+ current (Hodgkin-Huxley formulation, m^3*h*j gating)
    - ICaL : L-type Ca2+ current (Goldman-Hodgkin-Katz flux equation, d*f*q gating)
    - IK1  : Inward rectifier K+ current (instantaneous rectification)
    - IKr  : Rapid delayed rectifier K+ current (XKr gating + rectification)
    - IKs  : Slow delayed rectifier K+ current (XKs^2 gating)
    - Ito  : Transient outward K+ current (Xto*Yto gating)
    - IKp  : Plateau K+ current (instantaneous voltage dependence)
    - INCX : Na+/Ca2+ exchanger current

Calcium handling:
    - SERCA uptake (Hill equation, coefficient 2)
    - SR Ca2+ release via piecewise-linear Q function (Sato et al.)
    - Rapid-equilibrium Ca2+ buffering (SR sites, calmodulin, troponin)

Stochastic gating:
    When N_CaL > 0, the ICaL gates (d, f, q) are updated using a Langevin
    (channel noise) approach. Gaussian random variates are generated via the
    Box-Muller transform from uniform variates produced by a xorshift32 PRNG.
    The noise magnitude scales as 1/sqrt(N_CaL), so large channel populations
    approach deterministic behaviour.

Integration method:
    Forward Euler with dt = 0.05 ms (default).

References:
    Sato D, Bers DM. "How does stochastic ryanodine receptor-mediated
    Ca leak fail to initiate a Ca spark?" Biophys J. 2011.

See also:
    cell.h, cell.cc — original C++ implementation
"""

import math
import numpy as np


def xorshift(state):
    """Xorshift32 pseudo-random number generator (Marsaglia, 2003).

    Produces a full-period sequence of 2^32 - 1 values using three
    XOR-shift operations. This matches the C++ implementation exactly
    (same shift constants: 13, 17, 5) so that both versions produce
    identical random sequences from the same seed.

    Args:
        state: Current 32-bit unsigned integer state.

    Returns:
        Updated 32-bit unsigned integer state (also the next random value).
    """
    x = state & 0xFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17) & 0xFFFFFFFF
    x ^= (x << 5) & 0xFFFFFFFF
    return x


class Cell:
    """Sato-Bers ventricular cardiac myocyte model.

    This class encapsulates the 15 state variables and all biophysical
    parameters for a single ventricular myocyte. The pace() method advances
    the model by one Forward Euler time step (dt = 0.05 ms by default).

    State variables are stored in a NumPy array (self.x) and exposed as
    Python properties for convenient named access (self.v, self.ci, etc.).

    Attributes:
        x (np.ndarray): Array of 15 state variables.
        tauf (float): f-gate (ICaL voltage inactivation) time constant (ms).
        av (float): Slope parameter for piecewise-linear SR release Q function.
        gam (float): Exponent for Ca2+-dependent inactivation (q-gate).
        N_CaL (int): Number of L-type Ca2+ channels (0 for deterministic mode).
        xsx (int): xorshift32 PRNG state for stochastic gating.
    """

    # --- Constants ---
    VC = -70.0            # Voltage threshold for APD calculation (mV)
    STIM = 50.0           # Stimulus current amplitude (uA/uF)
    STIM_DURATION = 1.0   # Stimulus duration (ms)

    # Physical constants
    F = 96490.0           # Faraday's constant (C/mol)
    F2 = 96.490           # Faraday's constant (C/mmol)
    R = 8.315             # Ideal gas constant (J/mol·K)
    TEMP = 308.0          # Temperature (K)

    # Ionic concentrations (mM)
    K_IN = 149.4          # Intracellular K+
    NA_OUT = 140.0        # Extracellular Na+
    K_OUT = 4.0           # Extracellular K+
    CA_OUT_MM = 1.8       # Extracellular Ca2+

    N = 15                # Number of state variables

    UINT32_MAX = 0xFFFFFFFF

    def __init__(self):
        """Initialise the cell to resting-state conditions.

        Sets all 15 state variables to physiological resting values and
        configures default model parameters (conductances, time constants,
        SERCA parameters, stochastic gating settings) matching the C++
        constructor in cell.cc.

        State variable indices:
            0:  v    — Membrane potential (mV)
            1:  ci   — Bulk cytosolic Ca2+ (uM)
            2:  cs   — Subspace (dyadic cleft) Ca2+ (uM)
            3:  cj   — Network SR Ca2+ (uM)
            4:  cjp  — Junctional SR Ca2+ (uM)
            5:  Ir   — SR Ca2+ release current (uM/ms)
            6:  f    — ICaL voltage inactivation gate (0-1)
            7:  q    — ICaL Ca2+-dependent inactivation gate (0-1)
            8:  d    — ICaL activation gate (0-1)
            9:  h    — INa fast inactivation gate (0-1)
            10: j    — INa slow inactivation gate (0-1)
            11: XKr  — IKr activation gate (0-1)
            12: XKs  — IKs activation gate (0-1)
            13: Xto  — Ito activation gate (0-1)
            14: Yto  — Ito inactivation gate (0-1)
        """
        # State variables: [v, ci, cs, cj, cjp, Ir, f, q, d, h, j, XKr, XKs, Xto, Yto]
        self.x = np.zeros(self.N)
        self.x[0] = -90.0       # v
        self.x[1] = 0.5         # ci
        self.x[2] = 0.5         # cs
        self.x[3] = 115.0       # cj
        self.x[4] = 115.0       # cjp
        self.x[5] = 0.0         # Ir
        self.x[6] = 0.1         # f
        self.x[7] = 0.4         # q
        self.x[8] = 0.001       # d
        self.x[9] = 0.99869     # h
        self.x[10] = 0.99887    # j
        self.x[11] = 0.0        # XKr
        self.x[12] = 0.0001     # XKs
        self.x[13] = 3.742e-5   # Xto
        self.x[14] = 1.0        # Yto

        # Model parameters
        self.tauf = 52.0        # f-gate time constant
        self.av = 3.0           # u (Ca2+ release parameter)
        self.gam = 0.7          # gamma (Ca2+-dependent inactivation)
        self.taujj = 1.0        # j-gate tau factor
        self.dt = 0.05          # Time step (ms)

        # SERCA parameters
        self.cup = 0.5          # 0.5 uM
        self.vup = 250 * 0.001  # 250 uM/s

        # [Na]i
        self.Na_in = 10.0       # mM

        # Conductances
        self.gna = 12.0         # INa
        self.gk1 = 2.8          # IK1
        self.gkr = 0.0136       # IKr
        self.gks = 0.0245       # IKs
        self.gto = 0.1          # Ito
        self.gnaca = 6.0        # NCX
        self.gkp = 0.002216     # IKp
        self.icabar = 3.5       # ICaL

        # Stochastic gating
        self.N_CaL = 100000     # Number of L-type Ca channels
        self.xsx = 1821800813   # Xorshift RNG seed

    # --- Property accessors for state variables ---
    @property
    def v(self):
        return self.x[0]

    @v.setter
    def v(self, val):
        self.x[0] = val

    @property
    def ci(self):
        return self.x[1]

    @ci.setter
    def ci(self, val):
        self.x[1] = val

    @property
    def cs(self):
        return self.x[2]

    @cs.setter
    def cs(self, val):
        self.x[2] = val

    @property
    def cj(self):
        return self.x[3]

    @cj.setter
    def cj(self, val):
        self.x[3] = val

    @property
    def cjp(self):
        return self.x[4]

    @cjp.setter
    def cjp(self, val):
        self.x[4] = val

    @property
    def Ir(self):
        return self.x[5]

    @Ir.setter
    def Ir(self, val):
        self.x[5] = val

    @property
    def f(self):
        return self.x[6]

    @f.setter
    def f(self, val):
        self.x[6] = val

    @property
    def q(self):
        return self.x[7]

    @q.setter
    def q(self, val):
        self.x[7] = val

    @property
    def d(self):
        return self.x[8]

    @d.setter
    def d(self, val):
        self.x[8] = val

    @property
    def h(self):
        return self.x[9]

    @h.setter
    def h(self, val):
        self.x[9] = val

    @property
    def j(self):
        return self.x[10]

    @j.setter
    def j(self, val):
        self.x[10] = val

    @property
    def XKr(self):
        return self.x[11]

    @XKr.setter
    def XKr(self, val):
        self.x[11] = val

    @property
    def XKs(self):
        return self.x[12]

    @XKs.setter
    def XKs(self, val):
        self.x[12] = val

    @property
    def Xto(self):
        return self.x[13]

    @Xto.setter
    def Xto(self, val):
        self.x[13] = val

    @property
    def Yto(self):
        return self.x[14]

    @Yto.setter
    def Yto(self, val):
        self.x[14] = val

    @property
    def cnsr(self):
        return self.x[3]

    @cnsr.setter
    def cnsr(self, val):
        self.x[3] = val

    def pace(self, st=0.0):
        """Advance the cell model by one Forward Euler time step (dt).

        This method computes all ionic currents, calcium fluxes, and gating
        variable derivatives, then updates the 15 state variables in place.
        It is a direct port of CCell::pace() in cell.cc.

        Args:
            st (float): External stimulus current (uA/uF). Non-zero during
                the stimulus window, zero otherwise. Default is 0.0.

        Side effects:
            Updates all state variables (self.x) and the PRNG state (self.xsx)
            when stochastic gating is enabled.
        """
        # Cache state variables as local scalars for performance
        v = self.v
        ci = self.ci
        cs = self.cs
        cj = self.cj
        cjp = self.cjp
        Ir = self.Ir
        f = self.f
        q = self.q
        d = self.d
        h = self.h
        j = self.j
        XKr = self.XKr
        XKs = self.XKs
        Xto = self.Xto
        Yto = self.Yto

        dt = self.dt

        # ===================================================================
        # Fast Sodium Current (INa) — Hodgkin-Huxley formulation
        #
        # INa = gna * m^3 * h * j * (V - E_Na)
        #
        # m gate uses steady-state approximation (minf) because activation
        # is much faster than dt. h (fast inactivation) and j (slow
        # inactivation) are integrated via Forward Euler.
        # ===================================================================
        am = 0.32 * (v + 47.13) / (1 - math.exp(-0.1 * (v + 47.13)))  # m forward rate
        bm = 0.08 * math.exp(-v / 11)                                   # m backward rate
        minf = am / (am + bm)                                           # Steady-state m

        ah = 0.135 * math.exp((v + 80) / (-6.8))    # h forward rate
        bh = 7.5 / (1 + math.exp(-0.1 * (v + 11)))  # h backward rate
        dh = ah * (1 - h) - bh * h                   # dh/dt

        aj = 0.175 * math.exp((v + 100) / (-23)) / (1 + math.exp(0.15 * (v + 79)))  # j forward
        bj = 0.3 / (1 + math.exp(-0.1 * (v + 32)))                                   # j backward
        dj = (aj * (1 - j) - bj * j) / self.taujj                                     # dj/dt

        # Nernst equilibrium potential for Na+: E_Na = (RT/F) * ln([Na]_o / [Na]_i)
        ena = self.R * self.TEMP / self.F2 * math.log(self.NA_OUT / self.Na_in)
        INa = self.gna * minf**3 * h * j * (v - ena)

        # ===================================================================
        # Calcium Buffering — Rapid equilibrium approximation
        #
        # bci and bcs are buffering factors (0-1) that scale the effective
        # rate of change of free [Ca2+]. Three buffers are modeled:
        #   SR membrane sites (Bsr/Ksr), calmodulin (Bcd/Kcd), troponin (BT/KT)
        # beta = 1 / (1 + sum_i B_i * K_i / (Ca + K_i)^2)
        # ===================================================================
        Bsr = 47.0    # SR buffer total concentration (uM)
        Ksr = 0.6     # SR buffer dissociation constant (uM)
        Bcd = 24.0
        Kcd = 7.0
        k_on = 37.7 * 0.001
        k_off = 19.6 * 0.001
        BT = 70.0
        KT = k_off / k_on
        bci = 1 / (1 + Bsr * Ksr / (ci + Ksr)**2
                    + Bcd * Kcd / (ci + Kcd)**2
                    + BT * KT / (ci + KT)**2)
        bcs = 1 / (1 + Bsr * Ksr / (cs + Ksr)**2
                    + Bcd * Kcd / (cs + Kcd)**2
                    + BT * KT / (cs + KT)**2)

        # ===================================================================
        # L-type Calcium Current (ICaL) — Goldman-Hodgkin-Katz flux equation
        #
        # ICaL uses the GHK constant-field equation because Ca2+ concentrations
        # differ by orders of magnitude across the membrane. Gated by d*f*q.
        # Voltage is converted mV->V (v/1000), Ca2+ from uM->mM (cs/1000).
        # ===================================================================
        Pca = 2.7 / 3.5 * 0.00000054  # Ca2+ permeability (cm/s)
        vF_RT = (v / 1000) * self.F / self.R / self.TEMP
        jca = (d * f * q * self.icabar * Pca
               * (4 * (v / 1000) * self.F * self.F / self.R / self.TEMP)
               * ((cs / 1000) * math.exp(2 * vF_RT) - 0.341 * self.CA_OUT_MM)
               / (math.exp(2 * vF_RT) - 1))

        # ===================================================================
        # ICaL Gating Variable Updates — Deterministic or Stochastic
        #
        # Steady-state values:
        #   dinf — Boltzmann activation (half-activation ~5 mV)
        #   finf — Boltzmann inactivation (half-inactivation ~-35 mV)
        #   qinf — Hill-type Ca2+-dependent inactivation (subspace [Ca2+])
        # ===================================================================
        dinf = 1 / (1 + math.exp(-(v - 5.0) / 6.24))  # d-gate steady state
        taud = 5.0                                       # d-gate time constant (ms)
        finf = 1 / (1 + math.exp((v + 35.0) / 8.6))   # f-gate steady state
        cst = 1.0                                        # Reference [Ca2+] for q-gate (uM)
        qinf = 1 / (1 + (cs / cst)**self.gam)          # q-gate steady state
        tauq = 20.0                                      # q-gate time constant (ms)

        if self.N_CaL == 0:
            # ----- Deterministic mode: simple first-order relaxation -----
            # dx/dt = (x_inf - x) / tau_x  =>  x += dx * dt
            d += (dinf - d) / taud * dt
            f += (finf - f) / self.tauf * dt
            q += (qinf - q) / tauq * dt
        else:
            # ----- Stochastic mode: Langevin channel noise -----
            #
            # For a two-state gate x with N_CaL channels:
            #   dx = [alpha*(1-x) - beta*x]*dt + sqrt(2*A*dt) * Z
            # where:
            #   alpha = x_inf / tau,  beta = (1 - x_inf) / tau
            #   A = (alpha*(1-x) + beta*x) / N_CaL  (diffusion coefficient)
            #   Z ~ N(0,1) from Box-Muller transform
            #
            # Box-Muller transform: given U1, U2 ~ Uniform(0,1),
            #   Z1 = sqrt(-2*ln(U1)) * cos(2*pi*U2)
            #   Z2 = sqrt(-2*ln(U1)) * sin(2*pi*U2)
            # are independent standard normal variates.

            # Generate first pair of Gaussian variates via Box-Muller
            self.xsx = xorshift(self.xsx)
            ua1 = (float(self.xsx) + 1.0) / (float(self.UINT32_MAX) + 2.0)  # Uniform (0,1)
            self.xsx = xorshift(self.xsx)
            ua2 = (float(self.xsx) + 1.0) / (float(self.UINT32_MAX) + 2.0)  # Uniform (0,1)
            mag = math.sqrt(-2.0 * math.log(ua1))        # Rayleigh magnitude
            za1 = mag * math.cos(2.0 * math.pi * ua2)    # Gaussian variate 1
            za2 = mag * math.sin(2.0 * math.pi * ua2)    # Gaussian variate 2

            # d gate (activation) — Langevin update
            d_plus = dinf / taud                           # Opening rate
            d_minus = (1 - dinf) / taud                    # Closing rate
            Ad = (d_plus * (1 - d) + d_minus * d) / self.N_CaL  # Diffusion coeff
            dd = (d_plus * (1 - d) - d_minus * d) * dt + math.sqrt(2 * Ad * dt) * za1
            d = max(0.0, min(1.0, d + dd))                # Clamp to [0, 1]

            # f gate (voltage inactivation) — Langevin update
            f_plus = finf / self.tauf
            f_minus = (1 - finf) / self.tauf
            Af = (f_plus * (1 - f) + f_minus * f) / self.N_CaL
            df = (f_plus * (1 - f) - f_minus * f) * dt + math.sqrt(2 * Af * dt) * za2
            f = max(0.0, min(1.0, f + df))

            # Generate second pair of Gaussian variates (only za1 used for q)
            self.xsx = xorshift(self.xsx)
            ua1 = (float(self.xsx) + 1.0) / (float(self.UINT32_MAX) + 2.0)
            self.xsx = xorshift(self.xsx)
            ua2 = (float(self.xsx) + 1.0) / (float(self.UINT32_MAX) + 2.0)
            mag = math.sqrt(-2.0 * math.log(ua1))
            za1 = mag * math.cos(2.0 * math.pi * ua2)

            # q gate (Ca2+-dependent inactivation) — Langevin update
            q_plus = qinf / tauq
            q_minus = (1 - qinf) / tauq
            Aq = (q_plus * (1 - q) + q_minus * q) / self.N_CaL
            dq = (q_plus * (1 - q) - q_minus * q) * dt + math.sqrt(2 * Aq * dt) * za1
            q = max(0.0, min(1.0, q + dq))

        # ===================================================================
        # Sodium-Calcium Exchanger (NCX / INCX)
        #
        # Exchanges 3 Na+ for 1 Ca2+. The current depends on both Na+ and
        # Ca2+ concentrations with an exponential voltage dependence.
        # xi = 0.35 partitions voltage dependence between forward/reverse.
        # ===================================================================
        Kmna = 87.5    # Half-saturation for Na+ (mM)
        Kmca = 1.380   # Half-saturation for Ca2+ (mM)
        xi = 0.35      # Voltage partition coefficient
        ksat = 0.1     # Saturation factor at negative potentials
        a = (v / 1000) * self.F / self.R / self.TEMP  # Dimensionless voltage V*F/(R*T)
        jnaca = (self.gnaca
                 / (Kmna**3 + self.NA_OUT**3)
                 / (Kmca + self.CA_OUT_MM)
                 * (math.exp(xi * a) * self.Na_in**3 * self.CA_OUT_MM
                    - math.exp((xi - 1) * a) * self.NA_OUT**3 * (cs / 1000))
                 / (1 + ksat * math.exp((xi - 1) * a)))

        # ===================================================================
        # SERCA (SR Ca2+-ATPase) Uptake — Hill equation (coefficient 2)
        # jup = vup * ci^2 / (ci^2 + cup^2)
        # ===================================================================
        jup = self.vup * ci**2 / (ci**2 + self.cup**2)

        # ===================================================================
        # SR Ca2+ Release — Piecewise-linear Q function (Sato et al.)
        #
        # Q depends on junctional SR load (cjp):
        #   Q = 0                    if cjp < 50 uM  (below threshold)
        #   Q = 1.5 * (cjp - 50)    if 50 <= cjp < 110  (linear ramp)
        #   Q = av * cjp + w         if cjp >= 110  (steeper slope)
        # w ensures continuity at cjp = 110.
        # ===================================================================
        w = 1.5 * (110 - 50) - self.av * 110  # Continuity offset
        if cjp < 50:
            Q = 0.0
        elif cjp >= 110:
            Q = self.av * cjp + w
        else:
            Q = 1.5 * (cjp - 50)
        Q *= 0.001  # -> ms

        g = 250 * 0.3
        taur = 20.0
        dIr = -g * jca * Q / self.icabar - Ir / taur

        # ===================================================================
        # Inward Rectifier Potassium Current (IK1)
        # Stabilises resting potential near E_K with strong inward rectification.
        # ===================================================================
        EK = self.R * self.TEMP / self.F2 * math.log(self.K_OUT / self.K_IN)  # Nernst E_K
        KmK1 = 13.0
        k1inf = 1 / (2 + math.exp(1.62 * self.F2 / self.R / self.TEMP * (v - EK)))
        IK1 = self.gk1 * k1inf * self.K_OUT / (self.K_OUT + KmK1) * (v - EK)

        # ===================================================================
        # Rapid Delayed Rectifier Potassium Current (IKr)
        # Contributes to phase 3 repolarisation. Rv = rectification factor.
        # ===================================================================
        Rv = 1 / (1 + 2.5 * math.exp(0.1 * (v + 28)))
        IKr = self.gkr * Rv * XKr * math.sqrt(self.K_OUT / 4) * (v - EK)
        XKrinf = 1 / (1 + math.exp(-2.182 - 0.1819 * v))
        tauKr = 43 + 1 / (math.exp(-5.495 + 0.1691 * v) + math.exp(-7.677 - 0.0128 * v))
        dXKr = (XKrinf - XKr) / tauKr

        # ===================================================================
        # Slow Delayed Rectifier Potassium Current (IKs)
        # Slow activation, mixed K+/Na+ reversal potential (P_Na/P_K = 0.01833).
        # ===================================================================
        EKs = self.R * self.TEMP / self.F2 * math.log(
            (self.K_OUT + 0.01833 * self.NA_OUT) / (self.K_IN + 0.01833 * self.Na_in))
        IKs = self.gks * XKs**2 * (v - EKs)
        XKsinf = 1 / (1 + math.exp((v - 16) / (-13.6)))
        tauKs = 1 / ((0.0000719 * (v - 10)) / (1 - math.exp(-0.148 * (v - 10)))
                      + (0.000131 * (v - 10)) / (math.exp(0.0687 * (v - 10)) - 1))
        dXKs = (XKsinf - XKs) / tauKs

        # ===================================================================
        # Transient Outward Potassium Current (Ito)
        # Produces the phase 1 repolarisation notch. Xto = activation, Yto = inactivation.
        # ===================================================================
        Ito = self.gto * Xto * Yto * (v - EK)
        aXto = 0.04516 * math.exp(0.03577 * v)
        bXto = 0.0989 * math.exp(-0.06237 * v)
        dXto = aXto * (1 - Xto) - bXto * Xto
        aYto = 0.005415 * math.exp((v + 33.5) / (-5)) / (1 + 0.051335 * math.exp((v + 33.5) / (-5)))
        bYto = 0.005415 * math.exp((v + 33.5) / 5) / (1 + 0.051335 * math.exp((v + 33.5) / 5))
        dYto = aYto * (1 - Yto) - bYto * Yto

        # ===================================================================
        # Plateau Potassium Current (IKp)
        # Small, time-independent K+ current active during the AP plateau.
        # ===================================================================
        KKp = 1 / (1 + math.exp((7.488 - v) / 5.98))
        IKp = self.gkp * KKp * (v - EK)

        # ===================================================================
        # Calcium Concentration ODEs
        #
        # ci  — bulk cytosol: diffusion from subspace minus SERCA uptake
        # cs  — subspace: SR release + NCX - ICaL - diffusion to cytosol
        # cjp — junctional SR: refills from network SR
        # cj  — network SR: filled by SERCA, depleted by release
        #
        # vivs = cytosol/subspace volume ratio; taus = diffusion time (ms)
        # ===================================================================
        vivs = 10.0   # Cytosol-to-subspace volume ratio
        taus = 2.0    # Subspace-to-cytosol diffusion time constant (ms)
        dci = bci * ((cs - ci) / taus - jup)                           # d[Ca]_i/dt
        dcs = bcs * (vivs * (Ir - (cs - ci) / taus - jca + jnaca))    # d[Ca]_ss/dt
        taua = 50.0   # NSR-to-jSR transfer time constant (ms)
        dcjp = (cj - cjp) / taua                                       # d[Ca]_jSR/dt
        dcj = -Ir + jup                                                 # d[Ca]_NSR/dt

        # ===================================================================
        # Membrane Potential ODE
        # dV/dt = -(I_ion - I_stim)
        # Ca2+ fluxes converted to current by factor 0.02 * 1000 (uM->mM).
        # Factor of 2 on jca reflects the 2+ valence of Ca2+.
        # ===================================================================
        dv = -(-st + INa + IK1 + IKr + IKs + Ito + IKp + 0.02 * (jnaca + 2 * jca) * 1000)

        # ===================================================================
        # Forward Euler update: x(t+dt) = x(t) + dx/dt * dt
        # Note: d, f, q gates were already updated above.
        # ===================================================================
        self.v = v + dv * dt
        self.ci = ci + dci * dt
        self.cs = cs + dcs * dt
        self.cj = cj + dcj * dt
        self.cjp = cjp + dcjp * dt
        self.Ir = Ir + dIr * dt
        self.h = h + dh * dt
        self.j = j + dj * dt
        self.d = d  # already updated in stochastic/deterministic block
        self.f = f
        self.q = q
        self.XKr = XKr + dXKr * dt
        self.XKs = XKs + dXKs * dt
        self.Xto = Xto + dXto * dt
        self.Yto = Yto + dYto * dt
