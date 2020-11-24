#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class FLLP:
    def readinput(self, fname):
        # Create a numpy array to store coords
        coords = np.zeros((500, 2))
        idx = 0
        with open(fname) as f:
            for line in f:
                coords[idx][0] = float(line.split()[0])
                coords[idx][1] = float(line.split()[1])
                idx = idx + 1
        
        # Read and store values from the text file
        print(coords)
        
        # Calculate the distances and store in dist
        
        # Assign and f value
        return #dist, f
    
    def solve_lp(self):
        # Set up all the vaiables
        # Setup all the constraints
        # Solve the program
        return #sol
    
    def filter_and_round_solution(self, lsol):
        return
    
if __name__ == '__main__':
    flmip = FLLP()
    flmip.readinput('500_us_city_coords.txt')
    lsol = flmip.solve_lp()
    
    intsol = flmip.filter_and_round_solution(lsol)