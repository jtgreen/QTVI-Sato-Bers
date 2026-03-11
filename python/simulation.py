"""
Sato-Bers model: parameter sweep over tauf (20–60).

Measures APD (at -70 mV threshold) and peak intracellular calcium
for each tauf value over 10000 beats after 1000 beats of pre-pacing.
Output written to result.txt (tab-separated: tauf, APD, cimax).
"""

from cell import Cell


def main():
    tauff = 20.0
    u = 3.0

    cell = Cell()
    cell.tauf = tauff
    cell.av = u
    dt = cell.dt
    stim = Cell.STIM
    vc = Cell.VC

    with open("result.txt", "w") as os:
        for tf in range(20, 61):
            print(f"tauff = {tf}")

            tauff = float(tf)
            cell.tauf = tauff
            bcl = 300.0
            itr = 1000
            Tn = int(itr * bcl / dt)
            BCLn = int(bcl / dt)
            Durn = int(cell.STIM_DURATION / dt)

            # Pre-pacing to steady state
            for tn in range(Tn):
                if tn % BCLn < Durn:
                    cell.pace(stim)
                else:
                    cell.pace()

            # Measurement phase
            itr = 10000
            first = False
            cimax = 0.0
            apd = 0.0
            vold = cell.v
            APDt1 = 0.0
            Tn = int(itr * bcl / dt)

            for tn in range(Tn):
                t = tn * dt

                if tn % BCLn < Durn:
                    if first:
                        first = False
                        os.write(f"{tauff}\t{apd}\t{cimax}\n")
                        cimax = 0.0
                        apd = 0.0
                    cell.pace(stim)
                else:
                    first = True
                    cell.pace()

                if cimax < cell.ci:
                    cimax = cell.ci

                if vold < vc and cell.v > vc:
                    APDt1 = (t - dt) + dt * (vc - vold) / (cell.v - vold)
                elif vold > vc and cell.v < vc:
                    APDt2 = (t - dt) + dt * (vc - vold) / (cell.v - vold)
                    apd = APDt2 - APDt1

                vold = cell.v


if __name__ == "__main__":
    main()
