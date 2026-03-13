/*******************************************************************************
 * cell.cc — Implementation of the Sato-Bers ventricular cardiac myocyte model
 *
 * This file contains:
 *   1. Static constant definitions (physical constants, ionic concentrations)
 *   2. xorshift32 PRNG used for stochastic channel gating
 *   3. CCell constructor — initialises all 15 state variables to resting values
 *      and sets default model parameters (conductances, time constants, etc.)
 *   4. CCell destructor and assignment operator
 *   5. CCell::pace() — the core time-stepping routine that:
 *        a) Computes all ionic currents using Hodgkin-Huxley-type gating
 *        b) Evaluates calcium handling (buffering, SERCA, SR release, NCX)
 *        c) Optionally applies Langevin stochastic noise to ICaL gates
 *        d) Updates all state variables via Forward Euler integration
 *
 * Integration method: Forward Euler with dt = 0.05 ms (default).
 *
 * References:
 *   Sato D, Bers DM. "How does stochastic ryanodine receptor-mediated
 *   Ca leak fail to initiate a Ca spark?" Biophys J. 2011.
 ******************************************************************************/

#include "cell.h"

// --- Static constants initialization ---
const double CCell::vc = -70;         // Voltage threshold for APD calculation
const double CCell::stim = 50;        // Stimulus amplitude
const double CCell::stimduration = 1; // Stimulus duration in ms

// Physical constants
const double CCell::F = 96490;   // Faraday's constant (C/mol)
const double CCell::F2 = 96.490; // Faraday's constant (C/mmol)
const double CCell::R = 8.315;   // Ideal gas constant (J/mol·K)
const double CCell::Temp = 308;  // Temperature (K)

// Ionic concentrations
const double CCell::K_in = 149.4;   // Intracellular K+ (mM)
const double CCell::Na_out = 140;   // Extracellular Na+ (mM)
const double CCell::K_out = 4;      // Extracellular K+ (mM)
const double CCell::Ca_outmM = 1.8; // Extracellular Ca2+ (mM)

// Number of state variables
const int CCell::N = 15;

// ---------------------------------------------------------------------------
// xorshift32 pseudo-random number generator (Marsaglia, 2003).
// Produces a full-period sequence of 2^32 - 1 values with just three
// XOR-shift operations. Used to generate uniform random variates for
// the Box-Muller transform in the stochastic gating section of pace().
// ---------------------------------------------------------------------------
inline uint32_t xorshift(uint32_t *xx) {
  *xx ^= *xx << 13;
  *xx ^= *xx >> 17;
  *xx ^= *xx << 5;
  return *xx;
}

// ---------------------------------------------------------------------------
// Constructor: Allocate state variable array and initialise to resting-state
// values. Named reference members (v, ci, cs, ...) are bound to entries in
// x[] via the initialiser list. Default parameter values reproduce the
// standard Sato-Bers ventricular myocyte configuration.
// ---------------------------------------------------------------------------
CCell::CCell(void) : x(new double[15]),
                     v(x[0]), ci(x[1]), cs(x[2]), cj(x[3]), cjp(x[4]), Ir(x[5]), f(x[6]),
                     q(x[7]), d(x[8]), h(x[9]), j(x[10]), XKr(x[11]),
                     XKs(x[12]), Xto(x[13]), Yto(x[14]), cnsr(x[3]) {
  x[0] = -90.0;     // v;
  x[1] = 0.5;       // ci;
  x[2] = 0.5;       // cs;
  x[3] = 115.0;     // cj;
  x[4] = 115.0;     // cjp;
  x[5] = 0;         // Ir;
  x[6] = 0.1;       // f;
  x[7] = 0.4;       // q;
  x[8] = 0.001;     // d;
  x[9] = 0.99869;   // hNa;
  x[10] = 0.99887;  // jNa;
  x[11] = 0.0;      // XKr;
  x[12] = 0.0001;   // XKs;
  x[13] = 3.742e-5; // Xto;
  x[14] = 1.0;      // Yto;

  tauf = 52; // tauf
  av = 3.0;  // u
  gam = 0.7; // gamma
  taujj = 1; // j gate tau factor
  dt = 0.05; // time step 0.05 ms

  // SERCA parameters
  cup = 0.5;         // 0.5uM
  vup = 250 * 0.001; // 250uM/s

  //[Na]i
  Na_in = 10; // mM

  // conductances
  gna = 12.0;     // INa (fast Na current)
  gk1 = 2.8;      // IK1 (Inward rectifier K+ current)
  gkr = 0.0136;   // IKr (Rapid component of the delayed rectifier K+ current)
  gks = 0.0245;   // IKs (Slow component of the delayed rectifier K+ current)
  gto = 0.1;      // Ito (Transient outward K+ current)
  gnaca = 6;      // Na-Ca exchanger
  gkp = 0.002216; // IKp (Plateau K+ current)
  icabar = 3.5;   // ICaL

  N_CaL = 100000;   // Number of LTCC
  xsx = 1821800813; // Initial seed for xorshift RNG
}
// Destructor: free the dynamically allocated state variable array.
CCell::~CCell() {
  delete[] x;
}

// ---------------------------------------------------------------------------
// Assignment operator: deep-copy all state variables and model parameters.
// Necessary because the default shallow copy would alias the x[] pointer.
// ---------------------------------------------------------------------------
CCell &CCell::operator=(const CCell &cell) {
  if (&cell != this) {
    for (int i = 0; i < N; i++) {
      x[i] = cell.x[i];
    }
    tauf = cell.tauf;
    av = cell.av;
    gam = cell.gam;
    taujj = cell.taujj;
    dt = cell.dt;

    cup = cell.cup;
    vup = cell.vup;

    Na_in = cell.Na_in;

    // conductances
    gna = cell.gna;       // INa (fast Na current)
    gk1 = cell.gk1;       // IK1 (Inward rectifier K+ current)
    gkr = cell.gkr;       // IKr (Rapid component of the delayed rectifier K+ current)
    gks = cell.gks;       // IKs (Slow component of the delayed rectifier K+ current)
    gto = cell.gto;       // Ito (Transient outward K+ current)
    gnaca = cell.gnaca;   // Na-Ca exchanger
    gkp = cell.gkp;       // IKp (Plateau K+ current)
    icabar = cell.icabar; // ICaL
    N_CaL = cell.N_CaL;   // Number of LTCC
    xsx = cell.xsx;       // xorshift RNG seed
  }
  return (*this);
}

// ---------------------------------------------------------------------------
// pace() — Advance the cell model by one time step (dt) using Forward Euler.
//
// Parameters:
//   st — external stimulus current (uA/uF). Non-zero during the stimulus
//        window, zero otherwise.
//
// This method computes all ionic currents, calcium fluxes, and gating
// variable derivatives, then updates the 15 state variables in place.
// ---------------------------------------------------------------------------
void CCell::pace(double st) {

  // =========================================================================
  // Fast Sodium Current (INa) — Hodgkin-Huxley formulation
  //
  // INa = gna * m^3 * h * j * (V - E_Na)
  //
  // The m gate uses a steady-state approximation (minf) because its
  // activation kinetics are much faster than the integration time step.
  // h (fast inactivation) and j (slow inactivation) are integrated.
  // =========================================================================
  double am = (0.32 * (v + 47.13) / (1 - exp(-0.1 * (v + 47.13)))); // m-gate forward rate
  double bm = (0.08 * exp(-v / 11));                                  // m-gate backward rate
  double minf = am / (am + bm);                                       // Steady-state m (instantaneous)

  double ah = (0.135 * exp((v + 80) / (-6.8)));                          // h-gate forward rate
  double bh = (7.5 / (1 + exp(-0.1 * (v + 11))));                        // h-gate backward rate
  double dh = ah * (1 - h) - bh * h;                                      // dh/dt

  double aj = (0.175 * exp((v + 100) / (-23))) / (1 + exp(0.15 * (v + 79))); // j-gate forward rate
  double bj = (0.3 / (1 + exp(-0.1 * (v + 32))));                            // j-gate backward rate
  double dj = (aj * (1 - j) - bj * j) / taujj;                               // dj/dt, scaled by taujj

  // Nernst equilibrium potential for Na+ (mV): E_Na = (RT/F) * ln([Na]_o / [Na]_i)
  double ena = R * Temp / F2 * log(Na_out / Na_in);
  // Total fast sodium current with cubic m-gate dependence
  double INa = gna * minf * minf * minf * h * j * (v - ena);

  // =========================================================================
  // Calcium Buffering — Rapid equilibrium (instantaneous) approximation
  //
  // bci and bcs are buffering factors (0–1) that scale the effective rate
  // of change of free [Ca2+]. They account for three buffers:
  //   - SR membrane sites (Bsr, Ksr)
  //   - Calmodulin/dye (Bcd, Kcd)
  //   - Troponin (BT, KT)
  // The factor is: beta = 1 / (1 + sum_i B_i * K_i / (Ca + K_i)^2)
  // =========================================================================
  const double Bsr = 47;             // SR buffer total concentration (uM)
  const double Ksr = 0.6;            // SR buffer dissociation constant (uM)
  const double Bcd = 24;             // Calmodulin/dye total concentration (uM)
  const double Kcd = 7;              // Calmodulin/dye dissociation constant (uM)
  const double k_on = 37.7 * 0.001;  // Troponin on-rate (1/(uM*ms))
  const double k_off = 19.6 * 0.001; // Troponin off-rate (1/ms)
  const double BT = 70;              // Troponin total concentration (uM)
  const double KT = k_off / k_on;    // Troponin equilibrium dissociation constant
  // Buffering factor for bulk cytosol
  double bci = 1 / (1 + Bsr * Ksr / (ci + Ksr) / (ci + Ksr) + Bcd * Kcd / (ci + Kcd) / (ci + Kcd) + BT * KT / (ci + KT) / (ci + KT));
  // Buffering factor for subspace (dyadic cleft)
  double bcs = 1 / (1 + Bsr * Ksr / (cs + Ksr) / (cs + Ksr) + Bcd * Kcd / (cs + Kcd) / (cs + Kcd) + BT * KT / (cs + KT) / (cs + KT));

  // =========================================================================
  // L-type Calcium Current (ICaL) — Goldman-Hodgkin-Katz (GHK) flux equation
  //
  // ICaL uses the GHK constant-field equation rather than an ohmic driving
  // force because Ca2+ concentrations differ by orders of magnitude across
  // the membrane. The current is gated by:
  //   d — voltage-dependent activation
  //   f — voltage-dependent inactivation (time constant = tauf)
  //   q — Ca2+-dependent inactivation (subspace [Ca2+])
  //
  // jca = d * f * q * icabar * Pca * GHK(V, [Ca]_ss, [Ca]_o)
  //
  // Note: voltage is converted from mV to V (v/1000) for the GHK equation,
  // and [Ca] from uM to mM (cs/1000) for consistency with Ca_outmM.
  // =========================================================================
  const double Pca = 2.7 / 3.5 * 0.00000054; // Ca2+ permeability (cm/s), scaled by icabar normalisation
  double jca = d * f * q * icabar * Pca * (4 * (v / 1000) * F * F / R / Temp) * ((cs / 1000) * exp(2 * (v / 1000) * F / R / Temp) - 0.341 * Ca_outmM) / (exp(2 * (v / 1000) * F / R / Temp) - 1);

  // =========================================================================
  // ICaL Gating Variable Updates — Deterministic or Stochastic
  //
  // Steady-state values and time constants for the three ICaL gates:
  //   dinf — Boltzmann activation (half-activation ~5 mV)
  //   finf — Boltzmann inactivation (half-inactivation ~-35 mV)
  //   qinf — Hill-type Ca2+-dependent inactivation using subspace [Ca2+]
  // =========================================================================
  double dinf = 1 / (1 + exp(-(v - 5.0) / 6.24));  // d-gate steady state
  const double taud = 5; // d-gate time constant (ms)
  double finf = (1 / (1 + exp((v + 35.0) / 8.6))); // f-gate steady state
  const double cst = 1.0; // Reference Ca2+ for q-gate (uM)
  double qinf = 1 / (1 + pow((cs / cst), gam));     // q-gate steady state
  const double tauq = 20; // q-gate time constant (ms)

  if (N_CaL == 0) {
    // ----- Deterministic mode: simple first-order relaxation -----
    // dx/dt = (x_inf - x) / tau_x  =>  x += dx * dt  (Forward Euler)
    double dd = (dinf - d) / taud;
    d += dd * dt;
    double df = (finf - f) / tauf;
    f += df * dt;
    double dq = (qinf - q) / tauq;
    q += dq * dt;
  } else {
    // ----- Stochastic mode: Langevin (Fokker-Planck) channel noise -----
    //
    // For a two-state gate x with N_CaL channels, the stochastic ODE is:
    //   dx = [alpha*(1-x) - beta*x]*dt + sqrt(2*A*dt) * Z
    // where:
    //   alpha = x_inf / tau,  beta = (1 - x_inf) / tau
    //   A = (alpha*(1-x) + beta*x) / N_CaL   (diffusion coefficient)
    //   Z ~ N(0,1) is a standard Gaussian variate
    //
    // Gaussian variates are generated via the Box-Muller transform:
    //   Z1 = sqrt(-2*ln(U1)) * cos(2*pi*U2)
    //   Z2 = sqrt(-2*ln(U1)) * sin(2*pi*U2)
    // where U1, U2 ~ Uniform(0,1) from the xorshift32 PRNG.
    //
    // After updating, gates are clamped to [0, 1].

    // Generate first pair of Gaussian variates (za1, za2) via Box-Muller
    xorshift(&xsx);
    double ua1 = (double(xsx) + 1.0) / (double(UINT32_MAX) + 2.0); // Uniform (0,1)
    xorshift(&xsx);
    double ua2 = (double(xsx) + 1.0) / (double(UINT32_MAX) + 2.0); // Uniform (0,1)
    double mag = sqrt(-2.0 * log(ua1));          // Rayleigh magnitude
    double za1 = mag * cos(2.0 * M_PI * ua2);    // Gaussian variate 1
    double za2 = mag * sin(2.0 * M_PI * ua2);    // Gaussian variate 2

    // --- d gate (activation) with Langevin noise ---
    double d_plus = dinf / taud;                   // Opening rate
    double d_minus = (1 - dinf) / taud;            // Closing rate
    double Ad = (d_plus * (1 - d) + d_minus * d) / N_CaL; // Diffusion coefficient
    double dd = (d_plus * (1 - d) - d_minus * d) * dt + sqrt(2 * Ad * dt) * za1; // Langevin step
    d += dd;
    if (d < 0) {
      d = 0;    // Clamp: gate probability cannot be negative
    } else if (d > 1) {
      d = 1;    // Clamp: gate probability cannot exceed 1
    }

    // --- f gate (voltage inactivation) with Langevin noise ---
    double f_plus = finf / tauf;
    double f_minus = (1 - finf) / tauf;
    double Af = (f_plus * (1 - f) + f_minus * f) / N_CaL;
    double df = (f_plus * (1 - f) - f_minus * f) * dt + sqrt(2 * Af * dt) * za2;
    f += df;
    if (f < 0) {
      f = 0;
    } else if (f > 1) {
      f = 1;
    }

    // Generate second pair of Gaussian variates (only za1 used for q gate)
    xorshift(&xsx);
    ua1 = (double(xsx) + 1.0) / (double(UINT32_MAX) + 2.0);
    xorshift(&xsx);
    ua2 = (double(xsx) + 1.0) / (double(UINT32_MAX) + 2.0);
    mag = sqrt(-2.0 * log(ua1));
    za1 = mag * cos(2.0 * M_PI * ua2);
    // za2 = mag * sin(2.0 * M_PI * ua2);  // Not needed; only one gate left

    // --- q gate (Ca2+-dependent inactivation) with Langevin noise ---
    double q_plus = qinf / tauq;
    double q_minus = (1 - qinf) / tauq;

    double Aq = (q_plus * (1 - q) + q_minus * q) / N_CaL;
    double dq = (q_plus * (1 - q) - q_minus * q) * dt + sqrt(2 * Aq * dt) * za1;

    q += dq;
    if (q < 0) {
      q = 0;
    } else if (q > 1) {
      q = 1;
    }
  }

  // =========================================================================
  // Sodium-Calcium Exchanger (NCX / INCX)
  //
  // Exchanges 3 Na+ for 1 Ca2+ across the sarcolemma. The current depends
  // on both Na+ and Ca2+ concentrations and membrane potential through an
  // exponential voltage dependence. xi = 0.35 partitions the voltage
  // dependence between the forward and reverse modes.
  //
  // jnaca = gnaca * [exp(xi*a)*Na_in^3*Ca_o - exp((xi-1)*a)*Na_o^3*Ca_ss]
  //         / [(Km_Na^3 + Na_o^3) * (Km_Ca + Ca_o) * (1 + ksat*exp(...))]
  // where a = V*F/(R*T) (dimensionless voltage).
  // =========================================================================
  const double Kmna = 87.5;  // Half-saturation for Na+ (mM)
  const double Kmca = 1.380; // Half-saturation for Ca2+ (mM)
  const double xi = 0.35;    // Voltage partition coefficient (dimensionless)
  const double ksat = 0.1;   // Saturation factor at negative potentials
  double a = (v / 1000) * F / R / Temp; // Dimensionless voltage: V*F/(R*T)
  double jnaca = gnaca / (Kmna * Kmna * Kmna + Na_out * Na_out * Na_out) / (Kmca + Ca_outmM) * (exp(xi * a) * Na_in * Na_in * Na_in * Ca_outmM - exp((xi - 1) * a) * Na_out * Na_out * Na_out * (cs / 1000)) / (1 + ksat * exp((xi - 1) * a));

  // =========================================================================
  // SERCA (SR Ca2+-ATPase) Uptake — Hill equation (Hill coefficient = 2)
  //
  // jup = vup * ci^2 / (ci^2 + cup^2)
  //
  // Pumps Ca2+ from the cytosol into the SR with a sigmoidal dependence
  // on cytosolic [Ca2+]. vup is the max rate; cup is the half-max [Ca2+].
  // =========================================================================
  double jup = vup * ci * ci / (ci * ci + cup * cup);

  // =========================================================================
  // Sarcoplasmic Reticulum (SR) Ca2+ Release — Piecewise-linear Q function
  // (Sato et al.)
  //
  // The SR release gain Q depends on junctional SR Ca2+ load (cjp):
  //   Q = 0                        if cjp < 50  uM  (below threshold)
  //   Q = 1.5 * (cjp - 50)         if 50 <= cjp < 110  (linear ramp)
  //   Q = av * cjp + w             if cjp >= 110  (steeper slope = av)
  //
  // w ensures continuity at cjp = 110. The parameter av ("u") controls
  // the slope in the high-load regime, affecting the propensity for
  // Ca2+ alternans and arrhythmias.
  //
  // The release current derivative includes:
  //   - A trigger term proportional to ICaL (calcium-induced calcium release)
  //   - An exponential decay term with time constant taur
  // =========================================================================
  double w = 1.5 * (110 - 50) - av * 110; // Offset ensuring continuity at cjp=110
  double Q;
  if (cjp < 50) {
    Q = 0;                    // No release below threshold
  } else if (cjp >= 110) {
    Q = av * cjp + w;         // High-load regime (steep, parameter-dependent)
  } else {
    Q = 1.5 * (cjp - 50);    // Mid-range linear ramp
  }
  Q *= (0.001); // Convert from 1/s to 1/ms
  const double g = 250 * 0.3;  // Release gain scaling factor (uM/ms)
  const double taur = 20;      // Release current decay time constant (ms)
  double dIr = (-g * jca * Q / icabar - Ir / taur); // dIr/dt: trigger + decay

  // =========================================================================
  // Inward Rectifier Potassium Current (IK1)
  //
  // IK1 stabilises the resting membrane potential near E_K. It has strong
  // inward rectification (reduced outward current at depolarised potentials)
  // described by k1inf. The [K+]_o dependence (K_out / (K_out + KmK1))
  // accounts for extracellular potassium sensitivity.
  // =========================================================================
  double EK = R * Temp / F2 * log(K_out / K_in); // Nernst potential for K+ (mV)
  const double KmK1 = 13; // Half-saturation [K+]_o for IK1 (mM)
  double k1inf = 1 / (2 + exp(1.62 * F2 / R / Temp * (v - EK))); // Rectification factor
  double IK1 = gk1 * k1inf * K_out / (K_out + KmK1) * (v - EK);

  // =========================================================================
  // Rapid Delayed Rectifier Potassium Current (IKr)
  //
  // IKr contributes to repolarisation during phase 3 of the action potential.
  // Rv is an instantaneous inward rectification factor. XKr is the
  // time-dependent activation gate. sqrt(K_out/4) accounts for the
  // [K+]_o-dependent conductance scaling (reference [K+]_o = 4 mM).
  // =========================================================================
  double Rv = 1 / (1 + 2.5 * exp(0.1 * (v + 28)));     // Rectification factor
  double IKr = gkr * Rv * XKr * sqrt(K_out / 4) * (v - EK);
  double XKrinf = 1 / (1 + exp(-2.182 - 0.1819 * v));   // Steady-state activation
  double tauKr = (43 + 1 / (exp(-5.495 + 0.1691 * v) + exp(-7.677 - 0.0128 * v))); // Time constant (ms)
  double dXKr = (XKrinf - XKr) / tauKr;                  // dXKr/dt

  // =========================================================================
  // Slow Delayed Rectifier Potassium Current (IKs)
  //
  // IKs has slow activation kinetics and contributes to late repolarisation.
  // Its reversal potential EKs includes a small Na+ permeability
  // (P_Na/P_K = 0.01833), making it a mixed K+/Na+ current.
  // The gate XKs is squared (two identical activation gates).
  // =========================================================================
  double EKs = R * Temp / F2 * log((K_out + 0.01833 * Na_out) / (K_in + 0.01833 * Na_in)); // Modified Nernst with Na+ permeability
  double IKs = gks * XKs * XKs * (v - EKs);
  double XKsinf = 1 / (1 + exp((v - 16) / (-13.6)));   // Steady-state activation
  double tauKs = (1 / ((0.0000719 * (v - 10)) / (1 - exp(-0.148 * (v - 10))) + (0.000131 * (v - 10)) / (exp(0.0687 * (v - 10)) - 1))); // Time constant (ms)
  double dXKs = (XKsinf - XKs) / tauKs;

  // =========================================================================
  // Transient Outward Potassium Current (Ito)
  //
  // Ito produces the early (phase 1) repolarisation notch in the AP.
  // It has both an activation gate (Xto) and an inactivation gate (Yto),
  // each described by Hodgkin-Huxley alpha/beta rate constants.
  // =========================================================================
  double Ito = gto * Xto * Yto * (v - EK);
  double aXto = 0.04516 * exp(0.03577 * v);   // Xto forward (opening) rate
  double bXto = 0.0989 * exp(-0.06237 * v);   // Xto backward (closing) rate
  double dXto = aXto * (1 - Xto) - bXto * Xto;
  double aYto = 0.005415 * exp((v + 33.5) / (-5)) / (1 + 0.051335 * exp((v + 33.5) / (-5))); // Yto forward
  double bYto = 0.005415 * exp((v + 33.5) / (5)) / (1 + 0.051335 * exp((v + 33.5) / (5)));   // Yto backward
  double dYto = aYto * (1 - Yto) - bYto * Yto;

  // =========================================================================
  // Plateau Potassium Current (IKp)
  //
  // A small, time-independent K+ current active during the plateau phase.
  // KKp is a sigmoidal voltage dependence (no time-dependent gating).
  // =========================================================================
  double KKp = 1 / (1 + exp((7.488 - v) / 5.98)); // Voltage-dependent factor
  double IKp = gkp * KKp * (v - EK);

  // =========================================================================
  // Calcium Concentration ODEs
  //
  // Four Ca2+ compartments are modeled:
  //   ci  — bulk cytosol: gains from subspace diffusion, loses to SERCA
  //   cs  — subspace (dyadic cleft): SR release, ICaL, NCX, diffusion to cytosol
  //   cjp — junctional SR: refills from network SR (cj) with time constant taua
  //   cj  — network SR: filled by SERCA, depleted by SR release
  //
  // vivs = V_cytosol / V_subspace volume ratio (dimensionless).
  // taus = diffusion time constant between subspace and cytosol (ms).
  // taua = transfer time constant from network SR to junctional SR (ms).
  // Buffering factors (bci, bcs) multiply the free [Ca2+] derivatives.
  // =========================================================================
  const double vivs = 10.;   // Cytosol-to-subspace volume ratio
  const double taus = 2;     // Subspace-to-cytosol diffusion time constant (ms)
  double dci = (bci * ((cs - ci) / taus - jup));                           // d[Ca]_i/dt
  double dcs = bcs * (vivs * (Ir - (cs - ci) / taus - jca + jnaca));      // d[Ca]_ss/dt
  const double taua = 50;    // NSR-to-jSR transfer time constant (ms)
  double dcjp = ((cj - cjp) / taua);                                       // d[Ca]_jSR/dt
  double dcj = (-Ir + jup);                                                // d[Ca]_NSR/dt

  // =========================================================================
  // Membrane Potential ODE
  //
  // dV/dt = -(I_ion - I_stim)
  //
  // I_ion is the total ionic current. The Ca2+ fluxes (jca, jnaca) are
  // converted to membrane current units by multiplying by the capacitive
  // surface-to-volume factor 0.02 and by 1000 (uM -> mM conversion).
  // The factor of 2 on jca reflects the 2+ valence of Ca2+ (2 charges
  // per ion crossing the membrane).
  // =========================================================================
  double dv = -(-st + INa + IK1 + IKr + IKs + Ito + IKp + 0.02 * (jnaca + 2 * jca) * 1000);

  // =========================================================================
  // Forward Euler State Variable Update
  //
  // x(t + dt) = x(t) + dx/dt * dt
  //
  // Note: The d, f, q gates were already updated above (either
  // deterministically or stochastically) and are not updated here again.
  // =========================================================================
  v += dv * dt;
  ci += dci * dt;
  cs += dcs * dt;
  cj += dcj * dt;
  cjp += dcjp * dt;
  Ir += dIr * dt;
  h += dh * dt;
  j += dj * dt;
  XKr += dXKr * dt;
  XKs += dXKs * dt;
  Xto += dXto * dt;
  Yto += dYto * dt;
}
