# QTVI-Sato-Bers

## Project Overview
Cardiac myocyte electrophysiology simulation based on the **Sato-Bers ventricular cell model**. The model simulates action potential dynamics, calcium cycling, and includes stochastic gating for L-type calcium channels (ICaL).

## Model Description
- **15 state variables**: v (membrane potential), ci/cs/cj/cjp (calcium concentrations), Ir (SR release), f/q/d (ICaL gating), h/j (INa gating), XKr/XKs/Xto/Yto (K+ current gating)
- **Ion currents**: INa, ICaL, IK1, IKr, IKs, Ito, IKp, INCX
- **Calcium handling**: SERCA uptake, SR release (piecewise Q function), NCX, buffering
- **Stochastic gating**: Langevin noise on d/f/q gates (Box-Muller + xorshift RNG) when N_CaL > 0
- **Integration**: Forward Euler with dt = 0.05 ms

## Code Structure

### Original C++ (`cell.h`, `cell.cc`, `0d.cc`)
- `CCell` class encapsulates all state variables and the `pace()` method
- `0d.cc` runs a parameter sweep over tauf (20–60), measuring APD and peak calcium

### Python (`python/`)
- `cell.py` — `Cell` class, direct port of C++ model
- `simulation.py` — Main simulation script (parameter sweep)

### Julia (`julia/`)
- `SatoBers.jl` — Module with cell model adapted for Thunderbolt.jl
- `simulation.jl` — Standalone simulation script

## Building & Running

### C++ (original)
```bash
make
./0d
```

### Python
```bash
cd python
python simulation.py
```

### Julia
```bash
cd julia
julia simulation.jl
```

## Key Parameters
- `tauf`: Time constant for f-gate (ICaL inactivation) — swept 20–60 in main sim
- `av` (u): Ca2+ release function parameter (default 3.0)
- `bcl`: Basic cycle length (300 ms)
- `N_CaL`: Number of L-type Ca channels for stochastic gating (100000; set 0 for deterministic)

## THOU SHALT NOT
- Edit ArmyHeart.jl
- Only make PRs within this repository
- code single cell in python and julia and compare to their c++
- code tissue in Julia only, using ArmyHeart, which extends Thunderbolt.jl
- Not stop until you have recreated their key figures