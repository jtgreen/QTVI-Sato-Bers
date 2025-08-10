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

// xorshift random number generator.
inline uint32_t xorshift(uint32_t *xx) {
  *xx ^= *xx << 13;
  *xx ^= *xx >> 17;
  *xx ^= *xx << 5;
  return *xx;
}

// initialization
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
CCell::~CCell() {
  delete[] x;
}
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

// Calc 1 time step
void CCell::pace(double st) {

  // --- Fast Sodium Current (INa) ---
  double am = (0.32 * (v + 47.13) / (1 - exp(-0.1 * (v + 47.13))));
  double bm = (0.08 * exp(-v / 11));
  double minf = am / (am + bm);

  double ah = (0.135 * exp((v + 80) / (-6.8)));
  double bh = (7.5 / (1 + exp(-0.1 * (v + 11))));
  double dh = ah * (1 - h) - bh * h;

  double aj = (0.175 * exp((v + 100) / (-23))) / (1 + exp(0.15 * (v + 79)));
  double bj = (0.3 / (1 + exp(-0.1 * (v + 32))));
  double dj = (aj * (1 - j) - bj * j) / taujj;

  double ena = R * Temp / F2 * log(Na_out / Na_in);
  double INa = gna * minf * minf * minf * h * j * (v - ena);

  // --- Calcium Buffering ---
  const double Bsr = 47;             // 47umol
  const double Ksr = 0.6;            // 0.6um
  const double Bcd = 24;             // 24um
  const double Kcd = 7;              // 7um
  const double k_on = 37.7 * 0.001;  // 37.7uMs
  const double k_off = 19.6 * 0.001; // 19.6s
  const double BT = 70;              // 70umol/l cytosol
  const double KT = k_off / k_on;
  double bci = 1 / (1 + Bsr * Ksr / (ci + Ksr) / (ci + Ksr) + Bcd * Kcd / (ci + Kcd) / (ci + Kcd) + BT * KT / (ci + KT) / (ci + KT));
  double bcs = 1 / (1 + Bsr * Ksr / (cs + Ksr) / (cs + Ksr) + Bcd * Kcd / (cs + Kcd) / (cs + Kcd) + BT * KT / (cs + KT) / (cs + KT));

  // --- L-type Calcium Current (ICaL) ---
  const double Pca = 2.7 / 3.5 * 0.00000054; // 2.7/3.5*0.00054cm/s
  double jca = d * f * q * icabar * Pca * (4 * (v / 1000) * F * F / R / Temp) * ((cs / 1000) * exp(2 * (v / 1000) * F / R / Temp) - 0.341 * Ca_outmM) / (exp(2 * (v / 1000) * F / R / Temp) - 1);

  // --- Stochastic gating for ICaL ---
  double dinf = 1 / (1 + exp(-(v - 5.0) / 6.24));
  const double taud = 5; // ms
  double finf = (1 / (1 + exp((v + 35.0) / 8.6)));
  const double cst = 1.0; // uM
  double qinf = 1 / (1 + pow((cs / cst), gam));
  const double tauq = 20; // ms
  if (N_CaL == 0) {
    double dd = (dinf - d) / taud;
    d += dd * dt;
    double df = (finf - f) / tauf;
    f += df * dt;
    double dq = (qinf - q) / tauq;
    q += dq * dt;
  } else {
    xorshift(&xsx);
    double ua1 = (double(xsx) + 1.0) / (double(UINT32_MAX) + 2.0);
    xorshift(&xsx);
    double ua2 = (double(xsx) + 1.0) / (double(UINT32_MAX) + 2.0);
    double mag = sqrt(-2.0 * log(ua1));
    double za1 = mag * cos(2.0 * M_PI * ua2);
    double za2 = mag * sin(2.0 * M_PI * ua2);

    double d_plus = dinf / taud;
    double d_minus = (1 - dinf) / taud;
    double Ad = (d_plus * (1 - d) + d_minus * d) / N_CaL;
    double dd = (d_plus * (1 - d) - d_minus * d) * dt + sqrt(2 * Ad * dt) * za1;
    d += dd;
    if (d < 0) {
      d = 0;
    } else if (d > 1) {
      d = 1;
    }

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

    xorshift(&xsx);
    ua1 = (double(xsx) + 1.0) / (double(UINT32_MAX) + 2.0);
    xorshift(&xsx);
    ua2 = (double(xsx) + 1.0) / (double(UINT32_MAX) + 2.0);
    mag = sqrt(-2.0 * log(ua1));
    za1 = mag * cos(2.0 * M_PI * ua2);
    // za2 = mag * sin(2.0 * M_PI * ua2);

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

  // --- Sodium-Calcium Exchanger (NCX) ---
  const double Kmna = 87.5;  // 87.5mM;
  const double Kmca = 1.380; // 1.38mM
  const double xi = 0.35;
  const double ksat = 0.1;
  double a = (v / 1000) * F / R / Temp;
  double jnaca = gnaca / (Kmna * Kmna * Kmna + Na_out * Na_out * Na_out) / (Kmca + Ca_outmM) * (exp(xi * a) * Na_in * Na_in * Na_in * Ca_outmM - exp((xi - 1) * a) * Na_out * Na_out * Na_out * (cs / 1000)) / (1 + ksat * exp((xi - 1) * a));

  // SERCA uptake
  double jup = vup * ci * ci / (ci * ci + cup * cup);

  // --- Sarcoplasmic Reticulum (SR) Ca2+ Release (Ir) ---
  double w = 1.5 * (110 - 50) - av * 110;
  double Q;
  if (cjp < 50) {
    Q = 0;
  } else if (cjp >= 110) {
    Q = av * cjp + w;
  } else {
    Q = 1.5 * (cjp - 50);
  }
  Q *= (0.001); //->ms
  const double g = 250 * 0.3;
  const double taur = 20; // 20ms
  double dIr = (-g * jca * Q / icabar - Ir / taur);

  // --- Inward Rectifier Potassium Current (IK1) ---
  double EK = R * Temp / F2 * log(K_out / K_in);
  const double KmK1 = 13; // mM
  double k1inf = 1 / (2 + exp(1.62 * F2 / R / Temp * (v - EK)));
  double IK1 = gk1 * k1inf * K_out / (K_out + KmK1) * (v - EK);

  // --- Rapid Delayed Rectifier Potassium Current (IKr) ---
  double Rv = 1 / (1 + 2.5 * exp(0.1 * (v + 28)));
  double IKr = gkr * Rv * XKr * sqrt(K_out / 4) * (v - EK);
  double XKrinf = 1 / (1 + exp(-2.182 - 0.1819 * v));
  double tauKr = (43 + 1 / (exp(-5.495 + 0.1691 * v) + exp(-7.677 - 0.0128 * v))); //[ms]
  double dXKr = (XKrinf - XKr) / tauKr;

  // --- Slow Delayed Rectifier Potassium Current (IKs) ---
  double EKs = R * Temp / F2 * log((K_out + 0.01833 * Na_out) / (K_in + 0.01833 * Na_in));
  double IKs = gks * XKs * XKs * (v - EKs);
  double XKsinf = 1 / (1 + exp((v - 16) / (-13.6)));
  double tauKs = (1 / ((0.0000719 * (v - 10)) / (1 - exp(-0.148 * (v - 10))) + (0.000131 * (v - 10)) / (exp(0.0687 * (v - 10)) - 1))); //[ms]
  double dXKs = (XKsinf - XKs) / tauKs;

  // --- Transient Outward Potassium Current (Ito) ---
  double Ito = gto * Xto * Yto * (v - EK);
  double aXto = 0.04516 * exp(0.03577 * v);
  double bXto = 0.0989 * exp(-0.06237 * v);
  double dXto = aXto * (1 - Xto) - bXto * Xto;
  double aYto = 0.005415 * exp((v + 33.5) / (-5)) / (1 + 0.051335 * exp((v + 33.5) / (-5)));
  double bYto = 0.005415 * exp((v + 33.5) / (5)) / (1 + 0.051335 * exp((v + 33.5) / (5)));
  double dYto = aYto * (1 - Yto) - bYto * Yto;

  // --- Plateau Potassium Current (IKp) ---
  double KKp = 1 / (1 + exp((7.488 - v) / 5.98));
  double IKp = gkp * KKp * (v - EK);

  // --- Calcium Fluxes and Concentration Updates ---
  const double vivs = 10.;
  const double taus = 2; // 2ms
  double dci = (bci * ((cs - ci) / taus - jup));
  double dcs = bcs * (vivs * (Ir - (cs - ci) / taus - jca + jnaca));
  const double taua = 50; // 50ms
  double dcjp = ((cj - cjp) / taua);
  double dcj = (-Ir + jup);

  // --- Membrane Potential Update ---
  // Sum of all currents (and external stimulus 'st') determines the change in voltage.
  double dv = -(-st + INa + IK1 + IKr + IKs + Ito + IKp + 0.02 * (jnaca + 2 * jca) * 1000);

  // --- Update State Variables using Forward Euler Method ---
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
