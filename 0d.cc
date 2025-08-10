#include <fstream>
#include <iostream>
using namespace std;
#include "cell.h"

int main(void) {

  // --- Model Parameters ---
  double tauff = 20;
  double u = 3.0;

  // --- Cell Initialization ---
  CCell cell;
  cell.tauf = tauff;
  cell.av = u;
  double dt = cell.getdt();
  double stim = cell.getstim();
  double vc = cell.getvc();

  // Open a file to store the results
  ofstream os("result.txt");

  // --- Main Simulation Loop ---
  for (int tf = 20; tf <= 60; tf++) {
    cout << "tauff = " << tf << endl;

    tauff = tf;
    cell.tauf = tauff;
    const double bcl = 300;
    int itr = 1000;
    int Tn = int(itr * bcl / dt);
    int BCLn = int(bcl / dt);
    int Durn = int(cell.getstimduration() / dt);

    // Pre-pacing the cell to reach a steady state
    for (int tn = 0; tn < Tn; tn++) {
      if (tn % BCLn < Durn) {
        cell.pace(stim); // Pace with stimulus
      } else {
        cell.pace(); // Pace without stimulus
      }
    }

    itr = 10000;
    bool first = false;
    double cimax = 0;
    double apd = 0;
    double vold = cell.v;
    double APDt1 = 0;
    Tn = int(itr * bcl / dt);
    for (int tn = 0; tn < Tn; tn++) {

      double t = tn * dt;
      if (tn % BCLn < Durn) {
        if (first) {
          first = false;
          os << tauff << "\t" << apd << "\t" << cimax << endl;
          cimax = 0;
          apd = 0;
        }
        cell.pace(stim);
      } else {
        first = true;
        cell.pace();
      }

      if (cimax < cell.ci) {
        cimax = cell.ci;
      }
      if (vold < vc && cell.v > vc) {
        APDt1 = (t - dt) + dt * (vc - vold) / (cell.v - vold);
      } else if (vold > vc && cell.v < vc) {
        double APDt2 = (t - dt) + dt * (vc - vold) / (cell.v - vold);
        apd = APDt2 - APDt1;
      }
      vold = cell.v;
    }
  }
  return 0;
}
