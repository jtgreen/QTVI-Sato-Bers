#ifndef ___CELL_H
#define ___CELL_H

#include <cstdlib>
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdint>

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

  double *x;
  double &v;    // Membrane potential
  double &ci;   // Intracellular Ca2+ concentration
  double &cs;   // Subspace Ca2+ concentration
  double &cj;   // SR Ca2+ concentration
  double &cjp;  // jSR Ca2+ concentration
  double &Ir;   // Ca2+ release current from SR
  double &f;    // L-type Ca2+ current inactivation gate
  double &q;    // Ca2+-dependent inactivation of ICaL
  double &d;    // L-type Ca2+ current activation gate
  double &h;    // gating variable (Fast Na+ current)
  double &j;    // gating variable (Fast Na+ current)
  double &XKr;  // gating variable (Rapid delayed rectifier K+ current)
  double &XKs;  // gating variable (Slow delayed rectifier K+ current)
  double &Xto;  // gating variable (Transient outward K+ current)
  double &Yto;  // gating variable (Transient outward K+ current)
  double &cnsr; // Reference to cj

  // --- Accessor Methods ---
  int getdim(void) { return N; };                        // Get the number of state variables
  double getdt(void) { return dt; };                     // Get the time step
  double getvc(void) { return vc; };                     // Get the voltage threshold for APD
  double getstim(void) { return stim; };                 // Get the stimulus current amplitude
  double getstimduration(void) { return stimduration; }; // Get the stimulus duration

  // model parameters
  double av;    // u, Parameter for Ca2+ release function
  double tauf;  // Time constant for f-gate (ICaL inactivation)
  double gam;   // dgree of  Ca2+-dependent inactivation (q-gate)
  double taujj; // Time constant scaling factor for the j-gate (INa inactivation)

  // SERCA parameters
  double cup;
  double vup;

  double Na_in; // Intracellular Sodium concentration (mM)

  // conductances
  double gna;    // INa (fast Na current)
  double gk1;    // IK1 (Inward rectifier K+ current)
  double gkr;    // IKr (Rapid component of the delayed rectifier K+ current)
  double gks;    // IKs (Slow component of the delayed rectifier K+ current)
  double gto;    // Ito (Transient outward K+ current)
  double gnaca;  // Na-Ca exchanger
  double gkp;    // IKp (Plateau K+ current)
  double icabar; // ICaL

  // Stochastic gating
  int N_CaL; // # of CaL channels

  uint32_t xsx; // Random number generator state
};
#endif /* ___CELL_H */
