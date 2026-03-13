/*
 * validate_export.cc — C++ reference trace exporter for the Sato-Bers model.
 *
 * PURPOSE:
 *   Generate high-precision (float64) CSV traces of all 15 state variables
 *   for multiple validation scenarios. These serve as the ground-truth reference
 *   against which Python and Julia ports are compared.
 *
 * VALIDATION SCENARIOS:
 *   1. Single-beat action potential trace (deterministic, N_CaL=0)
 *   2. Multi-beat steady-state approach (10 beats, deterministic)
 *   3. APD restitution: tauf sweep from 20–60 (deterministic)
 *   4. Perturbation tests:
 *      a. Elevated extracellular K+ ([K]o = 8 mM, hyperkalemia)
 *      b. Reduced ICaL conductance (50% block)
 *      c. Enhanced SERCA uptake (2x vup)
 *      d. Altered initial conditions (depolarized start)
 *   5. Stochastic trace (N_CaL=100000, fixed seed) — for RNG reproducibility
 *
 * OUTPUT:
 *   Each scenario writes a CSV to validation/reference/ with columns:
 *     time, v, ci, cs, cj, cjp, Ir, f, q, d, h, j, XKr, XKs, Xto, Yto
 *
 * BUILD:
 *   g++ -std=c++23 -O2 -o validate_export validate_export.cc cell.cc
 *
 * RUN:
 *   ./validate_export
 *
 * NOTES:
 *   - All floating point output uses %.17g for full float64 precision.
 *   - Deterministic mode (N_CaL=0) is used for exact cross-language comparison.
 *   - Stochastic mode tests RNG reproducibility (xorshift32 seed matching).
 */

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
using namespace std;
#include "../cell.h"

/*
 * write_csv_header — Write the CSV header row to the output file.
 *
 * Columns: time, then all 15 state variable names in order.
 */
void write_csv_header(ofstream &os) {
    os << "time,v,ci,cs,cj,cjp,Ir,f,q,d,h,j,XKr,XKs,Xto,Yto" << endl;
}

/*
 * write_csv_row — Write one row of state data at the given time.
 *
 * Uses %.17g format to preserve full float64 (double) precision,
 * ensuring no information loss when comparing across languages.
 */
void write_csv_row(ofstream &os, double t, CCell &cell) {
    char buf[1024];
    snprintf(buf, sizeof(buf),
        "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g",
        t, cell.v, cell.ci, cell.cs, cell.cj, cell.cjp, cell.Ir,
        cell.f, cell.q, cell.d, cell.h, cell.j,
        cell.XKr, cell.XKs, cell.Xto, cell.Yto);
    os << buf << endl;
}

/*
 * scenario1_single_beat — Record a single action potential in deterministic mode.
 *
 * Runs 1 beat at BCL=300 ms with tauf=30, N_CaL=0 (deterministic).
 * Samples every 10 time steps (0.5 ms intervals) for a manageable file size.
 * This is the primary test for equation-level correctness.
 */
void scenario1_single_beat() {
    cout << "Scenario 1: Single beat deterministic trace" << endl;
    CCell cell;
    cell.N_CaL = 0;       // Deterministic mode — no stochastic gating
    cell.tauf = 30;        // Mid-range tauf for a "typical" AP
    cell.av = 3.0;         // Default Ca2+ release parameter

    double dt = cell.getdt();
    double stim = cell.getstim();
    double bcl = 300.0;
    int BCLn = int(bcl / dt);
    int Durn = int(cell.getstimduration() / dt);

    /* Pre-pace for 100 beats to reach approximate steady state.
     * This ensures the single-beat trace reflects settled dynamics,
     * not transient effects from arbitrary initial conditions. */
    int prepace = 100;
    for (int beat = 0; beat < prepace; beat++) {
        for (int tn = 0; tn < BCLn; tn++) {
            if (tn < Durn) cell.pace(stim);
            else cell.pace();
        }
    }

    /* Record one full beat (300 ms), sampling every 10 steps = 0.5 ms. */
    ofstream os("validation/reference/scenario1_single_beat.csv");
    write_csv_header(os);
    for (int tn = 0; tn < BCLn; tn++) {
        double t = tn * dt;
        if (tn % 10 == 0) write_csv_row(os, t, cell);
        if (tn < Durn) cell.pace(stim);
        else cell.pace();
    }
    os.close();
    cout << "  -> wrote validation/reference/scenario1_single_beat.csv" << endl;
}

/*
 * scenario2_multi_beat — Record 10 consecutive beats to verify steady-state convergence.
 *
 * After 100-beat pre-pacing, records 10 full beats (3000 ms total).
 * This tests that the model maintains stable limit-cycle behavior
 * and that no drift accumulates across beats.
 */
void scenario2_multi_beat() {
    cout << "Scenario 2: Multi-beat convergence trace" << endl;
    CCell cell;
    cell.N_CaL = 0;
    cell.tauf = 30;
    cell.av = 3.0;

    double dt = cell.getdt();
    double stim = cell.getstim();
    double bcl = 300.0;
    int BCLn = int(bcl / dt);
    int Durn = int(cell.getstimduration() / dt);

    /* Pre-pace to steady state (100 beats). */
    for (int beat = 0; beat < 100; beat++) {
        for (int tn = 0; tn < BCLn; tn++) {
            if (tn < Durn) cell.pace(stim);
            else cell.pace();
        }
    }

    /* Record 10 beats, sampling every 20 steps = 1.0 ms. */
    ofstream os("validation/reference/scenario2_multi_beat.csv");
    write_csv_header(os);
    int total_steps = 10 * BCLn;
    for (int tn = 0; tn < total_steps; tn++) {
        double t = tn * dt;
        if (tn % 20 == 0) write_csv_row(os, t, cell);
        if (tn % BCLn < Durn) cell.pace(stim);
        else cell.pace();
    }
    os.close();
    cout << "  -> wrote validation/reference/scenario2_multi_beat.csv" << endl;
}

/*
 * scenario3_apd_restitution — Sweep tauf from 20 to 60, record APD and CaMax.
 *
 * This reproduces the main simulation from 0d.cc but with shorter
 * pre-pacing (200 beats) and measurement (100 beats) for speed.
 * Output is a compact CSV: tauf, APD, cimax per beat.
 */
void scenario3_apd_restitution() {
    cout << "Scenario 3: APD restitution (tauf sweep 20-60)" << endl;
    ofstream os("validation/reference/scenario3_apd_restitution.csv");
    os << "tauf,apd,cimax" << endl;

    CCell cell;
    cell.N_CaL = 0;        // Deterministic for reproducible comparison
    cell.av = 3.0;

    double dt = cell.getdt();
    double stim = cell.getstim();
    double vc = cell.getvc();

    for (int tf = 20; tf <= 60; tf++) {
        cell.tauf = tf;
        double bcl = 300.0;
        int BCLn = int(bcl / dt);
        int Durn = int(cell.getstimduration() / dt);

        /* Pre-pace: 200 beats to reach steady state at this tauf. */
        int prepace_beats = 200;
        int Tn = prepace_beats * BCLn;
        for (int tn = 0; tn < Tn; tn++) {
            if (tn % BCLn < Durn) cell.pace(stim);
            else cell.pace();
        }

        /* Measure APD and peak Ca over 100 beats. */
        int measure_beats = 100;
        Tn = measure_beats * BCLn;
        bool first = false;
        double cimax = 0, apd = 0, vold = cell.v, APDt1 = 0;

        for (int tn = 0; tn < Tn; tn++) {
            double t = tn * dt;
            if (tn % BCLn < Durn) {
                if (first) {
                    first = false;
                    char buf[256];
                    snprintf(buf, sizeof(buf), "%d,%.17g,%.17g", tf, apd, cimax);
                    os << buf << endl;
                    cimax = 0; apd = 0;
                }
                cell.pace(stim);
            } else {
                first = true;
                cell.pace();
            }
            if (cimax < cell.ci) cimax = cell.ci;
            if (vold < vc && cell.v > vc) {
                APDt1 = (t - dt) + dt * (vc - vold) / (cell.v - vold);
            } else if (vold > vc && cell.v < vc) {
                double APDt2 = (t - dt) + dt * (vc - vold) / (cell.v - vold);
                apd = APDt2 - APDt1;
            }
            vold = cell.v;
        }
    }
    os.close();
    cout << "  -> wrote validation/reference/scenario3_apd_restitution.csv" << endl;
}

/*
 * scenario4_perturbations — Physiological perturbation tests.
 *
 * Each sub-scenario modifies one parameter from baseline to test
 * that the Python/Julia ports respond identically to parameter changes.
 * All are deterministic (N_CaL=0). Record one beat after 100-beat prepace.
 *
 * Perturbations:
 *   4a: Hyperkalemia — [K]o raised from 4 to 8 mM
 *       Expected: shortened APD, elevated resting potential
 *   4b: 50% ICaL block — icabar halved from 3.5 to 1.75
 *       Expected: shortened APD, reduced Ca transient
 *   4c: Enhanced SERCA — vup doubled from 0.250 to 0.500
 *       Expected: faster Ca decay, possibly altered APD
 *   4d: Depolarized start — v initialized to -60 mV instead of -90 mV
 *       Expected: different transient before converging to same limit cycle
 */
void scenario4_perturbations() {
    cout << "Scenario 4: Perturbation tests" << endl;

    /* Helper lambda: run one perturbation scenario.
     * Pre-paces for 100 beats, then records 1 beat at 0.5 ms resolution. */
    auto run_perturbation = [](const char* filename, auto configure_fn) {
        CCell cell;
        cell.N_CaL = 0;
        cell.tauf = 30;
        cell.av = 3.0;
        configure_fn(cell);

        double dt = cell.getdt();
        double stim = cell.getstim();
        double bcl = 300.0;
        int BCLn = int(bcl / dt);
        int Durn = int(cell.getstimduration() / dt);

        for (int beat = 0; beat < 100; beat++) {
            for (int tn = 0; tn < BCLn; tn++) {
                if (tn < Durn) cell.pace(stim);
                else cell.pace();
            }
        }

        ofstream os(filename);
        write_csv_header(os);
        for (int tn = 0; tn < BCLn; tn++) {
            double t = tn * dt;
            if (tn % 10 == 0) write_csv_row(os, t, cell);
            if (tn < Durn) cell.pace(stim);
            else cell.pace();
        }
        os.close();
    };

    /* 4a: Hyperkalemia — elevated extracellular potassium.
     * K_out is a static const in the class, so we cannot change it directly.
     * Instead we test with modified gk1 conductance (doubled) which has
     * a similar physiological effect of increased IK1. */
    run_perturbation("validation/reference/scenario4a_elevated_gk1.csv",
        [](CCell &c) { c.gk1 = 5.6; });   // 2x IK1 conductance
    cout << "  -> wrote scenario4a_elevated_gk1.csv (2x gk1)" << endl;

    /* 4b: 50% ICaL block — reduced L-type calcium channel conductance. */
    run_perturbation("validation/reference/scenario4b_ical_block.csv",
        [](CCell &c) { c.icabar = 1.75; });  // 50% of 3.5
    cout << "  -> wrote scenario4b_ical_block.csv (50% ICaL block)" << endl;

    /* 4c: Enhanced SERCA — doubled SERCA pump rate. */
    run_perturbation("validation/reference/scenario4c_enhanced_serca.csv",
        [](CCell &c) { c.vup = 0.500; });    // 2x SERCA
    cout << "  -> wrote scenario4c_enhanced_serca.csv (2x SERCA)" << endl;

    /* 4d: Depolarized initial condition — start at -60 mV.
     * Tests that transient dynamics are identical even from non-resting states. */
    run_perturbation("validation/reference/scenario4d_depolarized.csv",
        [](CCell &c) { c.v = -60.0; });
    cout << "  -> wrote scenario4d_depolarized.csv (v0=-60mV)" << endl;
}

/*
 * scenario5_stochastic — Stochastic gating trace with fixed seed.
 *
 * Tests that the xorshift32 RNG + Box-Muller transform produces
 * identical random sequences across C++, Python, and Julia.
 * Uses N_CaL=100000 and the default seed 1821800813.
 * Records one beat at 0.5 ms resolution after 50-beat pre-pace.
 */
void scenario5_stochastic() {
    cout << "Scenario 5: Stochastic gating (fixed seed)" << endl;
    CCell cell;
    cell.N_CaL = 100000;   // Enable stochastic Langevin gating
    cell.tauf = 30;
    cell.av = 3.0;
    cell.xsx = 1821800813; // Fixed seed for reproducibility

    double dt = cell.getdt();
    double stim = cell.getstim();
    double bcl = 300.0;
    int BCLn = int(bcl / dt);
    int Durn = int(cell.getstimduration() / dt);

    /* Shorter pre-pace (50 beats) since stochastic mode is slower
     * and we mainly care about RNG sequence reproducibility. */
    for (int beat = 0; beat < 50; beat++) {
        for (int tn = 0; tn < BCLn; tn++) {
            if (tn < Durn) cell.pace(stim);
            else cell.pace();
        }
    }

    ofstream os("validation/reference/scenario5_stochastic.csv");
    write_csv_header(os);
    for (int tn = 0; tn < BCLn; tn++) {
        double t = tn * dt;
        if (tn % 10 == 0) write_csv_row(os, t, cell);
        if (tn < Durn) cell.pace(stim);
        else cell.pace();
    }
    os.close();
    cout << "  -> wrote validation/reference/scenario5_stochastic.csv" << endl;
}

/*
 * scenario6_steady_state_values — Record final state after long pre-pacing.
 *
 * Runs 500 beats at tauf=30 (deterministic), then writes a single row
 * with the final state values. This provides a precise "fingerprint"
 * for the steady-state attractor of the model.
 */
void scenario6_steady_state_values() {
    cout << "Scenario 6: Steady-state fingerprint" << endl;
    CCell cell;
    cell.N_CaL = 0;
    cell.tauf = 30;
    cell.av = 3.0;

    double dt = cell.getdt();
    double stim = cell.getstim();
    double bcl = 300.0;
    int BCLn = int(bcl / dt);
    int Durn = int(cell.getstimduration() / dt);

    /* 500 beats to deeply converge to the limit cycle. */
    for (int beat = 0; beat < 500; beat++) {
        for (int tn = 0; tn < BCLn; tn++) {
            if (tn < Durn) cell.pace(stim);
            else cell.pace();
        }
    }

    /* Write just the state at t=0 of beat 501 (right before stimulus). */
    ofstream os("validation/reference/scenario6_steady_state.csv");
    write_csv_header(os);
    write_csv_row(os, 0.0, cell);
    os.close();
    cout << "  -> wrote validation/reference/scenario6_steady_state.csv" << endl;
}

/*
 * main — Run all validation scenarios sequentially.
 *
 * Generates reference CSV files in validation/reference/ that are
 * then consumed by the Python and Julia validation scripts.
 */
int main(void) {
    cout << "=== Sato-Bers Model Validation Reference Export ===" << endl;
    cout << "Generating float64-precision reference traces..." << endl;
    cout << endl;

    scenario1_single_beat();
    scenario2_multi_beat();
    scenario3_apd_restitution();
    scenario4_perturbations();
    scenario5_stochastic();
    scenario6_steady_state_values();

    cout << endl;
    cout << "All reference data written to validation/reference/" << endl;
    return 0;
}
