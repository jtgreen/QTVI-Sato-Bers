/*******************************************************************************
 * cell.h — Header for the Sato-Bers ventricular cardiac myocyte model
 *
 * This file defines the CCell class which encapsulates the electrophysiology
 * of a single ventricular cardiac myocyte based on the Sato-Bers model.
 *
 * The model includes 15 state variables tracking:
 *   - Membrane potential (v)
 *   - Intracellular calcium concentrations (ci, cs, cj, cjp)
 *   - Sarcoplasmic reticulum (SR) Ca2+ release current (Ir)
 *   - L-type Ca2+ channel (ICaL) gating variables (d, f, q)
 *   - Fast sodium current (INa) gating variables (h, j)
 *   - Potassium current gating variables (XKr, XKs, Xto, Yto)
 *
 * Ion currents modeled:
 *   INa   — Fast sodium current (Hodgkin-Huxley formulation)
 *   ICaL  — L-type calcium current (Goldman-Hodgkin-Katz flux equation)
 *   IK1   — Inward rectifier potassium current
 *   IKr   — Rapid delayed rectifier potassium current
 *   IKs   — Slow delayed rectifier potassium current
 *   Ito   — Transient outward potassium current
 *   IKp   — Plateau potassium current
 *   INCX  — Sodium-calcium exchanger current
 *
 * Calcium handling includes SERCA uptake, SR release with a piecewise-linear
 * Q function, NCX exchange, and rapid buffering approximation.
 *
 * Stochastic gating: When N_CaL > 0, the ICaL gates (d, f, q) are updated
 * using a Langevin (channel noise) approach with xorshift32 PRNG and
 * Box-Muller transform for Gaussian random variates.
 *
 * Integration: Forward Euler with a default time step of dt = 0.05 ms.
 *
 * References:
 *   Sato D, Bers DM. "How does stochastic ryanodine receptor-mediated
 *   Ca leak fail to initiate a Ca spark?" Biophys J. 2011.
 ******************************************************************************/

#ifndef ___CELL_H
#define ___CELL_H

#include <cstdlib>
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdint>

/**
 * CCell — Single ventricular cardiac myocyte in the Sato-Bers model.
 *
 * Holds the 15 state variables as a dynamically allocated array (x[0..14]),
 * with named reference members for convenient access (v, ci, cs, etc.).
 * The pace() method advances the model by one Forward Euler time step.
 */
class CCell {
private:
  // --- Constants for Simulation ---
  static const double vc;           // Voltage threshold for APD calculation (-70 mV)
  static const double stim;         // Stimulus current amplitude (50)
  static const double stimduration; // Duration of the stimulus (1 ms)

  // --- Physical Constants ---
  static const double F;    // Faraday's constant (96490 C/mol)
  static const double F2;   // Faraday's constant in different units (96.490 C/mmol)
  static const double R;    // Ideal gas constant (8.315 J/mol·K)
  static const double Temp; // Temperature (308 K, which is ~35°C)

  // --- Ionic Concentrations (mM) ---
  static const double K_in;     // Intracellular Potassium concentration (149.4 mM)
  static const double Na_out;   // Extracellular Sodium concentration (140 mM)
  static const double K_out;    // Extracellular Potassium concentration (4 mM)
  static const double Ca_outmM; // Extracellular Calcium concentration (1.8 mM)

  double dt; // Time step for the simulation (ms)

  static const int N; // Number of state variables (15)

public:
  // Applies a stimulus and computes one time step of the model
  void pace(double stim = 0);

  // Sets the time step dt
  void setdt(double dtt) { dt = dtt; };

  // Constructor
  CCell(void);

  // Destructor
  virtual ~CCell();

  // Assignment operator
  CCell &operator=(const CCell &cell);

  // -------------------------------------------------------------------------
  // State variable array and named references
  // The 15 state variables are stored contiguously in x[0..14].
  // Named reference members alias into this array for readability.
  // -------------------------------------------------------------------------
  double *x;      // Dynamically allocated array of 15 state variables
  double &v;      // x[0]  — Membrane potential (mV)
  double &ci;     // x[1]  — Bulk cytosolic (intracellular) Ca2+ concentration (uM)
  double &cs;     // x[2]  — Subspace (dyadic cleft) Ca2+ concentration (uM)
  double &cj;     // x[3]  — Network SR (NSR) Ca2+ concentration (uM)
  double &cjp;    // x[4]  — Junctional SR (jSR) Ca2+ concentration (uM)
  double &Ir;     // x[5]  — SR Ca2+ release current (uM/ms)
  double &f;      // x[6]  — ICaL voltage-dependent inactivation gate (0–1)
  double &q;      // x[7]  — ICaL Ca2+-dependent inactivation gate (0–1)
  double &d;      // x[8]  — ICaL activation gate (0–1)
  double &h;      // x[9]  — INa fast inactivation gate (0–1)
  double &j;      // x[10] — INa slow inactivation gate (0–1)
  double &XKr;    // x[11] — IKr activation gate (0–1)
  double &XKs;    // x[12] — IKs activation gate (0–1)
  double &Xto;    // x[13] — Ito activation gate (0–1)
  double &Yto;    // x[14] — Ito inactivation gate (0–1)
  double &cnsr;   // Alias for cj (x[3]) — network SR Ca2+ concentration

  // --- Accessor Methods ---
  int getdim(void) { return N; };                        // Get the number of state variables
  double getdt(void) { return dt; };                     // Get the time step
  double getvc(void) { return vc; };                     // Get the voltage threshold for APD
  double getstim(void) { return stim; };                 // Get the stimulus current amplitude
  double getstimduration(void) { return stimduration; }; // Get the stimulus duration

  // -------------------------------------------------------------------------
  // Model parameters — tuneable biophysical constants
  // -------------------------------------------------------------------------
  double av;    // "u" — slope of the piecewise-linear SR Ca2+ release Q function
  double tauf;  // Time constant for the f-gate (ICaL voltage inactivation, ms)
  double gam;   // Exponent (gamma) controlling Ca2+-dependent inactivation (q-gate)
  double taujj; // Scaling factor for the j-gate time constant (INa slow inactivation)

  // SERCA (SR Ca2+-ATPase) pump parameters
  double cup;   // Half-maximal Ca2+ concentration for SERCA uptake (uM)
  double vup;   // Maximum SERCA uptake rate (uM/ms)

  double Na_in; // Intracellular Sodium concentration (mM)

  // -------------------------------------------------------------------------
  // Maximum conductances / permeabilities for each ion current
  // Units: mS/uF (conductances) or cm/s (permeability for ICaL)
  // -------------------------------------------------------------------------
  double gna;    // Max conductance for INa (fast Na+ current)
  double gk1;    // Max conductance for IK1 (inward rectifier K+ current)
  double gkr;    // Max conductance for IKr (rapid delayed rectifier K+ current)
  double gks;    // Max conductance for IKs (slow delayed rectifier K+ current)
  double gto;    // Max conductance for Ito (transient outward K+ current)
  double gnaca;  // Scaling factor for the Na+/Ca2+ exchanger (NCX) current
  double gkp;    // Max conductance for IKp (plateau K+ current)
  double icabar; // Scaling factor for ICaL (L-type Ca2+ current permeability)

  // -------------------------------------------------------------------------
  // Stochastic gating parameters
  // When N_CaL > 0, Langevin noise is added to the ICaL gates (d, f, q).
  // The noise magnitude scales as 1/sqrt(N_CaL), so larger channel populations
  // approach deterministic behavior.
  // -------------------------------------------------------------------------
  int N_CaL;     // Number of L-type Ca2+ channels (0 = deterministic mode)

  uint32_t xsx;  // Internal state for the xorshift32 pseudo-random number generator
};
#endif /* ___CELL_H */
