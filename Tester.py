import os
import itertools
import cupy as np
import sys
from Maininjector import main

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please provide a sequence, e.g. python3 Tester.py QKLVFFAE")

    sequence = sys.argv[1]  # take sequence from command line
    
    gridshape = (200, 200, 60)  
    box_lengths = (225, 225) 
    max_iter = 2000  
    dx = 1.5

    rhop0_values = [0.11]
    vchi_ps_values = [0.3]
    vchi_pp_values = [0.2]
    bjerrum_values = [0.7]
    eps_yukawa_values = [0.03, 0.8]
    epsilon_hb_values = [0.5, 1.5]
    salt_fractions = [0.005]
    initial_gamma_values = [0.05]
    decay_yukawa = 5
    decay_es = 20

    param_combinations = list(itertools.product(
        rhop0_values, salt_fractions, vchi_ps_values, 
        vchi_pp_values, bjerrum_values, eps_yukawa_values, epsilon_hb_values
    ))
    print(f"\nTrying all parameter combinations for sequence: {sequence} ({len(param_combinations)} combinations)")

    for i, (rhop0, salt_fraction, vchi_ps, vchi_pp, bjerrum, eps_yukawa, epsilon_hb) in enumerate(param_combinations):
        gamma_success = None
        current_gamma_index = 0

        while current_gamma_index < len(initial_gamma_values):
            gamma = initial_gamma_values[current_gamma_index]
            outdir = f"{sequence}/bj{bjerrum}_vps{vchi_ps}_epsy{eps_yukawa}_epshb{epsilon_hb}/g{gamma}"
            os.makedirs(outdir, exist_ok=True)
            print(f"\n[{i+1}/{len(param_combinations)}] {sequence}, rhop0={rhop0}, salt={salt_fraction}, "
                  f"gamma={gamma}, eps_yukawa={eps_yukawa}, epsilon_hb={epsilon_hb}")

            results = main(sequence, epsilon_hb, vchi_pp, vchi_ps, eps_yukawa, decay_yukawa, bjerrum, decay_es, rhop0, max_iter, gamma, salt_fraction, gridshape, outdir)

            PartialsHystory, DeltaRhoHystory, LDVCs, LDVCmaxs, LDVCmins, Broken, PS = results

        if gamma_success is None:
            print(f"  -> All gamma values failed for this parameter set.")
