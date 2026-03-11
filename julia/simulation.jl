"""
Zero-dimensional (single-cell) simulation driver for the Sato-Bers model.

Julia port of the C++ driver `0d.cc`. This script performs a parameter sweep
over tauf (the ICaL f-gate inactivation time constant) from 20 ms to 60 ms.

For each tauf value:

1. **Pre-pacing phase** (1000 beats at BCL = 300 ms):
   The cell is paced to approach a periodic steady state so that transient
   startup effects from initial conditions do not contaminate measurements.

2. **Measurement phase** (10000 beats at BCL = 300 ms):
   On each beat, the following are recorded:
   - **APD**: Action potential duration at a threshold of -70 mV, computed
     via linear interpolation of the voltage crossing times.
   - **cimax**: Peak intracellular calcium concentration during the beat.

**Integration method**: Forward Euler with dt = 0.05 ms. The deterministic
RHS is computed by `cell_rhs_deterministic!`, and Langevin noise is applied
to the ICaL gates (d, f, q) by `stochastic_gate_update!`.

**Output**: Tab-separated file "result.txt" with columns: tauf, APD, cimax.

**APD Calculation**:
The upstroke time (t1) and repolarisation time (t2) are found by detecting
when the membrane potential crosses the threshold voltage (vc = -70 mV).
Linear interpolation between consecutive time steps gives sub-dt accuracy:
    t_cross = t_old + dt * (vc - V_old) / (V_new - V_old)
    APD = t2 - t1

References:
    Sato D, Bers DM. "How does stochastic ryanodine receptor-mediated
    Ca leak fail to initiate a Ca spark?" Biophys J. 2011.

See also: `0d.cc` (C++ original), `python/simulation.py` (Python port).
"""

include("SatoBers.jl")
using .SatoBers

"""
    main()

Run the tauf parameter sweep simulation. Sweeps tauf from 20 to 60 ms,
pre-paces each configuration for 1000 beats, then measures APD and peak
[Ca²⁺]_i over 10000 beats. Results are written to "result.txt".
"""
function main()
    # Initial model configuration
    model = SatoBersModel{Float64}(tauf=20.0, av=3.0)
    dt    = 0.05             # Forward Euler integration time step (ms)
    stim  = 50.0             # Stimulus current amplitude (µA/µF)
    vc    = -70.0            # APD voltage threshold (mV)
    stim_duration = 1.0      # Stimulus duration (ms)

    # Allocate mutable state vector (from default resting-state initial conditions)
    u  = collect(default_initial_state(model))
    du = zeros(15)           # Derivative vector (filled by cell_rhs_deterministic!)

    # Initialise the xorshift32 PRNG for stochastic gating (same seed as C++)
    rng = StochasticState(UInt32(1821800813))

    open("result.txt", "w") do io
        # =================================================================
        # Parameter Sweep: tauf from 20 to 60 ms
        #
        # Varying tauf modulates ICaL inactivation kinetics, which changes
        # action potential duration and calcium transient amplitude — key
        # determinants of alternans and arrhythmia susceptibility.
        # =================================================================
        for tf in 20:60
            println("tauff = $tf")

            # Reconstruct model with new tauf (struct is immutable)
            model = SatoBersModel{Float64}(
                tauf = Float64(tf),
                av   = 3.0,
            )

            bcl  = 300.0                          # Basic cycle length (ms)
            BCLn = round(Int, bcl / dt)           # Time steps per beat
            Durn = round(Int, stim_duration / dt) # Stimulus duration in time steps

            # =============================================================
            # Phase 1: Pre-pacing (1000 beats)
            #
            # The cell is paced repeatedly to reach a dynamical steady
            # state (or periodic orbit) before any measurements are taken.
            # This eliminates transient behaviour from initial conditions.
            # =============================================================
            itr = 1000
            Tn  = round(Int, itr * bcl / dt)      # Total pre-pacing time steps
            for tn in 0:Tn-1
                if mod(tn, BCLn) < Durn
                    cell_rhs_deterministic!(du, u, model, stim)  # Stimulus window
                else
                    cell_rhs_deterministic!(du, u, model, 0.0)   # No stimulus
                end
                # Apply Langevin noise to ICaL gates (d, f, q) if N_CaL > 0
                stochastic_gate_update!(u, du, dt, model, rng)
                # Forward Euler update: u(t+dt) = u(t) + du * dt
                @. u += du * dt
            end

            # =============================================================
            # Phase 2: Measurement (10000 beats)
            #
            # Record APD and peak Ca²⁺ for each beat. The 'first' flag
            # detects beat boundaries at the start of each stimulus window.
            # =============================================================
            itr    = 10000
            Tn     = round(Int, itr * bcl / dt)
            first  = false
            cimax  = 0.0      # Peak [Ca²⁺]_i within the current beat
            apd    = 0.0      # APD of the current beat (ms)
            vold   = u[1]     # Previous time step voltage (for crossing detection)
            APDt1  = 0.0      # Time of upstroke crossing

            for tn in 0:Tn-1
                t = tn * dt   # Current simulation time (ms)

                # Detect beat boundaries at the start of each stimulus window
                if mod(tn, BCLn) < Durn
                    if first
                        # New beat starting — output previous beat's measurements
                        first = false
                        println(io, "$tf\t$apd\t$cimax")
                        cimax = 0.0
                        apd   = 0.0
                    end
                    cell_rhs_deterministic!(du, u, model, stim)
                else
                    first = true   # Inside a beat (past the stimulus window)
                    cell_rhs_deterministic!(du, u, model, 0.0)
                end
                stochastic_gate_update!(u, du, dt, model, rng)
                @. u += du * dt

                # Track peak intracellular calcium
                if cimax < u[2]
                    cimax = u[2]
                end

                # ----- APD measurement via linear interpolation -----
                if vold < vc && u[1] > vc
                    # Upstroke crossing: voltage crosses vc from below
                    APDt1 = (t - dt) + dt * (vc - vold) / (u[1] - vold)
                elseif vold > vc && u[1] < vc
                    # Repolarisation crossing: voltage crosses vc from above
                    APDt2 = (t - dt) + dt * (vc - vold) / (u[1] - vold)
                    apd = APDt2 - APDt1  # APD = repolarisation time - upstroke time
                end
                vold = u[1]
            end
        end
    end
end

main()
