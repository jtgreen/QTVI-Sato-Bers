"""
Sato-Bers model: parameter sweep over tauf (20–60).

Standalone simulation matching the C++ 0d.cc driver.
Measures APD and peak intracellular calcium for each tauf value.
"""

include("SatoBers.jl")
using .SatoBers

function main()
    model = SatoBersModel{Float64}(tauf=20.0, av=3.0)
    dt    = 0.05
    stim  = 50.0
    vc    = -70.0
    stim_duration = 1.0

    u  = collect(default_initial_state(model))
    du = zeros(15)

    # Stochastic RNG state
    rng = StochasticState(UInt32(1821800813))

    open("result.txt", "w") do io
        for tf in 20:60
            println("tauff = $tf")

            # Update tauf in model (immutable struct, so reconstruct)
            model = SatoBersModel{Float64}(
                tauf = Float64(tf),
                av   = 3.0,
            )

            bcl  = 300.0
            BCLn = round(Int, bcl / dt)
            Durn = round(Int, stim_duration / dt)

            # --- Pre-pacing (1000 beats) ---
            itr = 1000
            Tn  = round(Int, itr * bcl / dt)
            for tn in 0:Tn-1
                if mod(tn, BCLn) < Durn
                    cell_rhs_deterministic!(du, u, model, stim)
                else
                    cell_rhs_deterministic!(du, u, model, 0.0)
                end
                stochastic_gate_update!(u, du, dt, model, rng)
                @. u += du * dt
            end

            # --- Measurement (10000 beats) ---
            itr    = 10000
            Tn     = round(Int, itr * bcl / dt)
            first  = false
            cimax  = 0.0
            apd    = 0.0
            vold   = u[1]
            APDt1  = 0.0

            for tn in 0:Tn-1
                t = tn * dt

                if mod(tn, BCLn) < Durn
                    if first
                        first = false
                        println(io, "$tf\t$apd\t$cimax")
                        cimax = 0.0
                        apd   = 0.0
                    end
                    cell_rhs_deterministic!(du, u, model, stim)
                else
                    first = true
                    cell_rhs_deterministic!(du, u, model, 0.0)
                end
                stochastic_gate_update!(u, du, dt, model, rng)
                @. u += du * dt

                if cimax < u[2]
                    cimax = u[2]
                end

                if vold < vc && u[1] > vc
                    APDt1 = (t - dt) + dt * (vc - vold) / (u[1] - vold)
                elseif vold > vc && u[1] < vc
                    APDt2 = (t - dt) + dt * (vc - vold) / (u[1] - vold)
                    apd = APDt2 - APDt1
                end
                vold = u[1]
            end
        end
    end
end

main()
