/*******************************************************************************
 * 0d.cc — Zero-dimensional (single-cell) simulation driver for the
 *          Sato-Bers ventricular cardiac myocyte model.
 *
 * This program performs a parameter sweep over tauf (the ICaL f-gate
 * inactivation time constant) from 20 ms to 60 ms. For each tauf value:
 *
 *   1. Pre-pacing phase (1000 beats at BCL = 300 ms):
 *      The cell is paced to approach a periodic steady state so that
 *      transient startup effects do not contaminate the measurements.
 *
 *   2. Measurement phase (10000 beats at BCL = 300 ms):
 *      On each beat, the following are recorded:
 *        - APD: Action potential duration measured at a threshold of -70 mV,
 *               computed via linear interpolation of the voltage crossing.
 *        - cimax: Peak intracellular calcium concentration during the beat.
 *
 * Output: Tab-separated file "result.txt" with columns:
 *         tauf   APD   cimax
 *
 * APD Calculation Detail:
 *   APD is measured between upstroke crossing (V crosses -70 mV upward)
 *   and repolarisation crossing (V crosses -70 mV downward).
 *   Linear interpolation between consecutive time steps is used to find
 *   the exact crossing times:
 *     t_cross = t_old + dt * (vc - V_old) / (V_new - V_old)
 *   where vc = -70 mV is the threshold voltage.
 *
 * References:
 *   Sato D, Bers DM. "How does stochastic ryanodine receptor-mediated
 *   Ca leak fail to initiate a Ca spark?" Biophys J. 2011.
 ******************************************************************************/

#include <fstream>
#include <iostream>
using namespace std;
#include "cell.h"

int main(void) {

  // --- Model Parameters ---
  double tauff = 20;   // Initial f-gate time constant (ms); will be swept 20–60
  double u = 3.0;      // Ca2+ release function slope parameter ("av")

  // --- Cell Initialization ---
  CCell cell;
  cell.tauf = tauff;
  cell.av = u;
  double dt = cell.getdt();              // Integration time step (0.05 ms)
  double stim = cell.getstim();          // Stimulus current amplitude (50 uA/uF)
  double vc = cell.getvc();              // APD voltage threshold (-70 mV)

  // Open a file to store the results (tab-separated: tauf, APD, cimax)
  ofstream os("result.txt");

  // =========================================================================
  // Main Parameter Sweep: tauf from 20 to 60 ms
  //
  // Varying tauf changes ICaL inactivation kinetics, which modulates the
  // action potential duration and calcium transient amplitude — key
  // determinants of alternans and arrhythmia susceptibility.
  // =========================================================================
  for (int tf = 20; tf <= 60; tf++) {
    cout << "tauff = " << tf << endl;

    tauff = tf;
    cell.tauf = tauff;
    const double bcl = 300;                          // Basic cycle length (ms)
    int itr = 1000;                                  // Number of pre-pacing beats
    int Tn = int(itr * bcl / dt);                    // Total pre-pacing time steps
    int BCLn = int(bcl / dt);                        // Time steps per beat
    int Durn = int(cell.getstimduration() / dt);     // Stimulus duration in time steps

    // -----------------------------------------------------------------
    // Phase 1: Pre-pacing (1000 beats)
    //
    // The cell is paced repeatedly to reach a dynamical steady state
    // (or periodic orbit) before any measurements are taken. This
    // eliminates transient behaviour from the initial conditions.
    // -----------------------------------------------------------------
    for (int tn = 0; tn < Tn; tn++) {
      if (tn % BCLn < Durn) {
        cell.pace(stim); // Apply stimulus current during the stimulus window
      } else {
        cell.pace();     // No stimulus outside the window
      }
    }

    // -----------------------------------------------------------------
    // Phase 2: Measurement (10000 beats)
    //
    // Record APD and peak Ca2+ for each beat. The 'first' flag detects
    // the transition from the no-stimulus period to the stimulus period,
    // indicating the start of a new beat — at which point the previous
    // beat's APD and cimax are written to the output file.
    // -----------------------------------------------------------------
    itr = 10000;
    bool first = false;
    double cimax = 0;       // Running maximum of [Ca2+]_i within the current beat
    double apd = 0;         // APD of the current beat (ms)
    double vold = cell.v;   // Voltage from the previous time step (for crossing detection)
    double APDt1 = 0;       // Time of the upstroke crossing (V crosses vc upward)
    Tn = int(itr * bcl / dt);
    for (int tn = 0; tn < Tn; tn++) {

      double t = tn * dt;   // Current simulation time (ms)

      // Detect beat boundaries: when the stimulus window begins
      if (tn % BCLn < Durn) {
        if (first) {
          // A new beat is starting — output measurements from the previous beat
          first = false;
          os << tauff << "\t" << apd << "\t" << cimax << endl;
          cimax = 0;
          apd = 0;
        }
        cell.pace(stim);
      } else {
        first = true;       // We are past the stimulus window (inside a beat)
        cell.pace();
      }

      // Track peak intracellular calcium during this beat
      if (cimax < cell.ci) {
        cimax = cell.ci;
      }

      // ----- APD measurement via linear interpolation -----
      // Upstroke crossing: voltage crosses vc from below
      if (vold < vc && cell.v > vc) {
        // Linearly interpolate the exact crossing time
        APDt1 = (t - dt) + dt * (vc - vold) / (cell.v - vold);
      }
      // Repolarisation crossing: voltage crosses vc from above
      else if (vold > vc && cell.v < vc) {
        double APDt2 = (t - dt) + dt * (vc - vold) / (cell.v - vold);
        apd = APDt2 - APDt1; // APD = time between upstroke and repolarisation crossings
      }
      vold = cell.v;
    }
  }
  return 0;
}
