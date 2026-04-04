# Conversion Progress: QTVI-Sato-Bers

## Goal
Convert the Sato-Bers cardiac myocyte model from C++ to Python and Julia.
The Julia version will be structured for use with **Thunderbolt.jl**.

## Status

### Phase 1: Python Conversion
- [x] Analyze C++ source (cell.h, cell.cc, 0d.cc)
- [x] Create `python/cell.py` — Cell class with all ion currents, Ca handling, stochastic gating
- [x] Create `python/simulation.py` — Parameter sweep (tauf 20–60), APD/CaMax measurement
- [ ] Validate Python output against C++ results

### Phase 2: Julia Conversion (Thunderbolt.jl)
- [x] Create `julia/SatoBers.jl` — Module with cell model
- [x] Create `julia/simulation.jl` — Standalone simulation
- [ ] Integrate with Thunderbolt.jl cell model interface
- [ ] Validate Julia output against C++/Python results

### Phase 3: Validation & Testing
- [ ] Run C++ reference simulation, save output
- [ ] Compare Python output (should match C++ exactly for deterministic mode)
- [ ] Compare Julia output
- [ ] Document any numerical differences (expected with stochastic mode)
  
### Phase 4: Comparison with "sato QTVI.pdf" figures
- [ ] Reproduce key figures from the paper (APD restitution, Ca handling, etc.)
- [ ] Single cell
- [ ] Tissue (note, ArmyHeart is a submodule and has the capacity to run cables and 2d tissue along w/ code for calculating the pseudo-ECG and QTVI)

## Notes
- Stochastic gating uses xorshift RNG + Box-Muller transform — must match seed behavior across languages for reproducibility
- Forward Euler with dt=0.05 ms — consider offering RK4 or adaptive methods in Julia
- Thunderbolt.jl integration TBD — need to understand their cell model interface
