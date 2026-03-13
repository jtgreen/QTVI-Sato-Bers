"""
Zero-dimensional (single-cell) simulation driver for the Sato-Bers model.

Python port of the C++ driver 0d.cc. This script performs a parameter sweep
over tauf (the ICaL f-gate inactivation time constant) from 20 ms to 60 ms.
For each tauf value it:

    1. Pre-paces the cell for 1000 beats (BCL = 300 ms) to reach a dynamical
       steady state, eliminating transient effects from initial conditions.

    2. Runs a measurement phase of 10000 beats, recording on each beat:
       - APD: Action potential duration at a threshold of -70 mV, determined
         by linear interpolation of the voltage crossing times.
       - cimax: Peak intracellular calcium concentration.

Output:
    Tab-separated file "result.txt" with columns: tauf, APD, cimax.

APD Calculation:
    The upstroke time (t1) and repolarisation time (t2) are found by detecting
    when the membrane potential crosses the threshold voltage (vc = -70 mV).
    Linear interpolation between consecutive time steps gives sub-dt accuracy:
        t_cross = t_old + dt * (vc - V_old) / (V_new - V_old)
    APD = t2 - t1.

References:
    Sato D, Bers DM. "How does stochastic ryanodine receptor-mediated
    Ca leak fail to initiate a Ca spark?" Biophys J. 2011.

See also:
    0d.cc — original C++ driver
    cell.py — Cell class implementation
"""

from cell import Cell


def main():
    """Run the tauf parameter sweep simulation.

    Sweeps tauf from 20 to 60 ms inclusive, pre-paces each configuration
    for 1000 beats, then measures APD and peak [Ca2+]_i over 10000 beats.
    Results are written to 'result.txt'.
    """
    tauff = 20.0   # Initial f-gate time constant (ms)
    u = 3.0        # Ca2+ release function slope parameter (av)

    # Initialise cell with default resting state
    cell = Cell()
    cell.tauf = tauff
    cell.av = u
    dt = cell.dt               # Integration time step (0.05 ms)
    stim = Cell.STIM           # Stimulus current amplitude (50 uA/uF)
    vc = Cell.VC               # APD threshold voltage (-70 mV)

    with open("result.txt", "w") as os:
        # =================================================================
        # Parameter Sweep: tauf from 20 to 60 ms
        #
        # Varying tauf modulates ICaL inactivation kinetics, which changes
        # action potential duration and calcium transient amplitude — key
        # determinants of alternans and arrhythmia susceptibility.
        # =================================================================
        for tf in range(20, 61):
            print(f"tauff = {tf}")

            tauff = float(tf)
            cell.tauf = tauff
            bcl = 300.0                           # Basic cycle length (ms)
            itr = 1000                            # Number of pre-pacing beats
            Tn = int(itr * bcl / dt)              # Total pre-pacing time steps
            BCLn = int(bcl / dt)                  # Time steps per beat
            Durn = int(cell.STIM_DURATION / dt)   # Stimulus duration in time steps

            # ----- Phase 1: Pre-pacing (1000 beats) -----
            # Pace to steady state to eliminate initial transients.
            for tn in range(Tn):
                if tn % BCLn < Durn:
                    cell.pace(stim)   # Stimulus window
                else:
                    cell.pace()       # No stimulus

            # ----- Phase 2: Measurement (10000 beats) -----
            itr = 10000
            first = False
            cimax = 0.0        # Peak [Ca2+]_i within the current beat
            apd = 0.0          # APD of the current beat (ms)
            vold = cell.v      # Previous time step voltage (for crossing detection)
            APDt1 = 0.0        # Time of upstroke crossing
            Tn = int(itr * bcl / dt)

            for tn in range(Tn):
                t = tn * dt    # Current simulation time (ms)

                # Detect beat boundaries at the start of each stimulus window
                if tn % BCLn < Durn:
                    if first:
                        # New beat starting — output previous beat's measurements
                        first = False
                        os.write(f"{tauff}\t{apd}\t{cimax}\n")
                        cimax = 0.0
                        apd = 0.0
                    cell.pace(stim)
                else:
                    first = True   # Inside a beat (past the stimulus window)
                    cell.pace()

                # Track peak intracellular calcium
                if cimax < cell.ci:
                    cimax = cell.ci

                # APD measurement via linear interpolation of voltage crossings
                if vold < vc and cell.v > vc:
                    # Upstroke crossing: V crosses vc from below
                    APDt1 = (t - dt) + dt * (vc - vold) / (cell.v - vold)
                elif vold > vc and cell.v < vc:
                    # Repolarisation crossing: V crosses vc from above
                    APDt2 = (t - dt) + dt * (vc - vold) / (cell.v - vold)
                    apd = APDt2 - APDt1

                vold = cell.v


if __name__ == "__main__":
    main()
