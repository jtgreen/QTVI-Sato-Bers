"""
Sato-Bers ventricular cardiac myocyte model.

Direct port from C++ implementation. Includes:
- Fast Na+ current (INa)
- L-type Ca2+ current (ICaL) with stochastic gating
- Inward rectifier K+ current (IK1)
- Rapid delayed rectifier K+ current (IKr)
- Slow delayed rectifier K+ current (IKs)
- Transient outward K+ current (Ito)
- Plateau K+ current (IKp)
- Na-Ca exchanger (INCX)
- SERCA uptake
- SR Ca2+ release
- Ca2+ buffering
"""

import math
import numpy as np


def xorshift(state):
    """Xorshift32 PRNG matching the C++ implementation."""
    x = state & 0xFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17) & 0xFFFFFFFF
    x ^= (x << 5) & 0xFFFFFFFF
    return x


class Cell:
    # --- Constants ---
    VC = -70.0            # Voltage threshold for APD calculation (mV)
    STIM = 50.0           # Stimulus current amplitude
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
        """Compute one time step of the model."""
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

        # --- Fast Sodium Current (INa) ---
        am = 0.32 * (v + 47.13) / (1 - math.exp(-0.1 * (v + 47.13)))
        bm = 0.08 * math.exp(-v / 11)
        minf = am / (am + bm)

        ah = 0.135 * math.exp((v + 80) / (-6.8))
        bh = 7.5 / (1 + math.exp(-0.1 * (v + 11)))
        dh = ah * (1 - h) - bh * h

        aj = 0.175 * math.exp((v + 100) / (-23)) / (1 + math.exp(0.15 * (v + 79)))
        bj = 0.3 / (1 + math.exp(-0.1 * (v + 32)))
        dj = (aj * (1 - j) - bj * j) / self.taujj

        ena = self.R * self.TEMP / self.F2 * math.log(self.NA_OUT / self.Na_in)
        INa = self.gna * minf**3 * h * j * (v - ena)

        # --- Calcium Buffering ---
        Bsr = 47.0
        Ksr = 0.6
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

        # --- L-type Calcium Current (ICaL) ---
        Pca = 2.7 / 3.5 * 0.00000054
        vF_RT = (v / 1000) * self.F / self.R / self.TEMP
        jca = (d * f * q * self.icabar * Pca
               * (4 * (v / 1000) * self.F * self.F / self.R / self.TEMP)
               * ((cs / 1000) * math.exp(2 * vF_RT) - 0.341 * self.CA_OUT_MM)
               / (math.exp(2 * vF_RT) - 1))

        # --- Stochastic / Deterministic gating for ICaL ---
        dinf = 1 / (1 + math.exp(-(v - 5.0) / 6.24))
        taud = 5.0
        finf = 1 / (1 + math.exp((v + 35.0) / 8.6))
        cst = 1.0
        qinf = 1 / (1 + (cs / cst)**self.gam)
        tauq = 20.0

        if self.N_CaL == 0:
            # Deterministic
            d += (dinf - d) / taud * dt
            f += (finf - f) / self.tauf * dt
            q += (qinf - q) / tauq * dt
        else:
            # Stochastic (Langevin + Box-Muller)
            self.xsx = xorshift(self.xsx)
            ua1 = (float(self.xsx) + 1.0) / (float(self.UINT32_MAX) + 2.0)
            self.xsx = xorshift(self.xsx)
            ua2 = (float(self.xsx) + 1.0) / (float(self.UINT32_MAX) + 2.0)
            mag = math.sqrt(-2.0 * math.log(ua1))
            za1 = mag * math.cos(2.0 * math.pi * ua2)
            za2 = mag * math.sin(2.0 * math.pi * ua2)

            # d gate
            d_plus = dinf / taud
            d_minus = (1 - dinf) / taud
            Ad = (d_plus * (1 - d) + d_minus * d) / self.N_CaL
            dd = (d_plus * (1 - d) - d_minus * d) * dt + math.sqrt(2 * Ad * dt) * za1
            d = max(0.0, min(1.0, d + dd))

            # f gate
            f_plus = finf / self.tauf
            f_minus = (1 - finf) / self.tauf
            Af = (f_plus * (1 - f) + f_minus * f) / self.N_CaL
            df = (f_plus * (1 - f) - f_minus * f) * dt + math.sqrt(2 * Af * dt) * za2
            f = max(0.0, min(1.0, f + df))

            # New random numbers for q gate
            self.xsx = xorshift(self.xsx)
            ua1 = (float(self.xsx) + 1.0) / (float(self.UINT32_MAX) + 2.0)
            self.xsx = xorshift(self.xsx)
            ua2 = (float(self.xsx) + 1.0) / (float(self.UINT32_MAX) + 2.0)
            mag = math.sqrt(-2.0 * math.log(ua1))
            za1 = mag * math.cos(2.0 * math.pi * ua2)

            # q gate
            q_plus = qinf / tauq
            q_minus = (1 - qinf) / tauq
            Aq = (q_plus * (1 - q) + q_minus * q) / self.N_CaL
            dq = (q_plus * (1 - q) - q_minus * q) * dt + math.sqrt(2 * Aq * dt) * za1
            q = max(0.0, min(1.0, q + dq))

        # --- Sodium-Calcium Exchanger (NCX) ---
        Kmna = 87.5
        Kmca = 1.380
        xi = 0.35
        ksat = 0.1
        a = (v / 1000) * self.F / self.R / self.TEMP
        jnaca = (self.gnaca
                 / (Kmna**3 + self.NA_OUT**3)
                 / (Kmca + self.CA_OUT_MM)
                 * (math.exp(xi * a) * self.Na_in**3 * self.CA_OUT_MM
                    - math.exp((xi - 1) * a) * self.NA_OUT**3 * (cs / 1000))
                 / (1 + ksat * math.exp((xi - 1) * a)))

        # --- SERCA uptake ---
        jup = self.vup * ci**2 / (ci**2 + self.cup**2)

        # --- SR Ca2+ Release ---
        w = 1.5 * (110 - 50) - self.av * 110
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

        # --- IK1 ---
        EK = self.R * self.TEMP / self.F2 * math.log(self.K_OUT / self.K_IN)
        KmK1 = 13.0
        k1inf = 1 / (2 + math.exp(1.62 * self.F2 / self.R / self.TEMP * (v - EK)))
        IK1 = self.gk1 * k1inf * self.K_OUT / (self.K_OUT + KmK1) * (v - EK)

        # --- IKr ---
        Rv = 1 / (1 + 2.5 * math.exp(0.1 * (v + 28)))
        IKr = self.gkr * Rv * XKr * math.sqrt(self.K_OUT / 4) * (v - EK)
        XKrinf = 1 / (1 + math.exp(-2.182 - 0.1819 * v))
        tauKr = 43 + 1 / (math.exp(-5.495 + 0.1691 * v) + math.exp(-7.677 - 0.0128 * v))
        dXKr = (XKrinf - XKr) / tauKr

        # --- IKs ---
        EKs = self.R * self.TEMP / self.F2 * math.log(
            (self.K_OUT + 0.01833 * self.NA_OUT) / (self.K_IN + 0.01833 * self.Na_in))
        IKs = self.gks * XKs**2 * (v - EKs)
        XKsinf = 1 / (1 + math.exp((v - 16) / (-13.6)))
        tauKs = 1 / ((0.0000719 * (v - 10)) / (1 - math.exp(-0.148 * (v - 10)))
                      + (0.000131 * (v - 10)) / (math.exp(0.0687 * (v - 10)) - 1))
        dXKs = (XKsinf - XKs) / tauKs

        # --- Ito ---
        Ito = self.gto * Xto * Yto * (v - EK)
        aXto = 0.04516 * math.exp(0.03577 * v)
        bXto = 0.0989 * math.exp(-0.06237 * v)
        dXto = aXto * (1 - Xto) - bXto * Xto
        aYto = 0.005415 * math.exp((v + 33.5) / (-5)) / (1 + 0.051335 * math.exp((v + 33.5) / (-5)))
        bYto = 0.005415 * math.exp((v + 33.5) / 5) / (1 + 0.051335 * math.exp((v + 33.5) / 5))
        dYto = aYto * (1 - Yto) - bYto * Yto

        # --- IKp ---
        KKp = 1 / (1 + math.exp((7.488 - v) / 5.98))
        IKp = self.gkp * KKp * (v - EK)

        # --- Calcium fluxes ---
        vivs = 10.0
        taus = 2.0
        dci = bci * ((cs - ci) / taus - jup)
        dcs = bcs * (vivs * (Ir - (cs - ci) / taus - jca + jnaca))
        taua = 50.0
        dcjp = (cj - cjp) / taua
        dcj = -Ir + jup

        # --- Membrane potential ---
        dv = -(-st + INa + IK1 + IKr + IKs + Ito + IKp + 0.02 * (jnaca + 2 * jca) * 1000)

        # --- Forward Euler update ---
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
