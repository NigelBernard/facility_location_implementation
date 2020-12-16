#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pickle
import psutil
import time
from datetime import timedelta
import platform
from ortools.linear_solver import pywraplp

class FLLP:    
    def readinput(self, fname, noc):
        # Create a numpy array to store coords
        # Read and store values from the text file
        coords = np.zeros((noc, 2))
        idx = 0
        with open(fname) as f:
            for line in f:
                coords[idx][0] = float(line.split()[0])
                coords[idx][1] = float(line.split()[1])
                idx = idx + 1

        # Calculate the distances and store in dist
        dist = np.zeros((noc, noc))
        for i in range(noc):
            for j in range(noc):
                dist[i][j] = np.linalg.norm(coords[i] - coords[j])

        # Assign and f value
        f = 1
        pickle.dump(dist.flatten(), open( 'distances.pkl', "wb" ))
        return dist.flatten(), f
    
    def solve_lp(self, dist, f, noc):        
        # Instantiate a solver
        solver = pywraplp.Solver('FL', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

        objective = solver.Objective()
        # Set up all the vaiables for the objective function
        # Forumla -> min Sum(f * y_i) + Sum(dij * xij)
        y = [[]] * noc
        for i in range(noc):
            string = 'y' + str(i)
            y[i] = solver.NumVar(0.0, 1.0, string)
            objective.SetCoefficient(y[i], f)
        
        nocsq = noc**2
        x = [[]] * nocsq
        for i in range(nocsq):
            x[i] = solver.NumVar(0.0, 1.0, 'x{}'.format(i))
            objective.SetCoefficient(x[i], dist[i])
                
        objective.SetMinimization()
            
        # Setup all the constraints
        # Sum of all x_ij for j \in V = 1
        constraints1 = [[]] * noc
        for k in range(noc):
            constraints1[k] = solver.Constraint(1, 1)
            for s in range(k, nocsq, noc):
                constraints1[k].SetCoefficient(x[s], 1)
        
        constraints2 = [[]] * nocsq
        s = -1
        for k in range(nocsq):
            if (k % noc == 0):
                s = s + 1
                
            constraints2[k] = solver.Constraint(-solver.infinity(), 0)
            constraints2[k].SetCoefficient(x[k], 1)
            constraints2[k].SetCoefficient(y[s], -1)
        
        # Solve the program
        print('Number of constraints', solver.NumConstraints())
        print('Number of variables', solver.NumVariables())
        print('Starting the solver...')
        solver.Solve()

        # Extract data from solver class
        yis = np.zeros(noc)
        xis = np.zeros(nocsq)
        for i in range(noc):
            yis[i] = y[i].solution_value()

        for i in range(nocsq):
            if (i % (noc + 1) != 0):
                xis[i] = x[i].solution_value()

        print('Optimal value =', objective.Value())
        facoc = sum(yis * f)
        totconc = sum(xis * dist)
        return yis, xis, facoc, totconc
    
    def filter_round_solution(self, y, x, dist, alpha = False):
        # Calculate delta j
        delj = np.zeros(y.size)
        noc = y.size
        nocsq = noc**2
        for idx in range(noc):
            idxsc = [x for x in range(idx, nocsq, noc)]
            delj[idx] = sum(dist[idxsc] * x[idxsc])
            
        # Calculate all the Bjs for each terminal
        B = [0] * noc
        if (alpha == False):
            # 6-approximation
            for i in range(noc):
                bset = set()
                for p in range(noc):
                    if (p != i and dist[i * noc + p] < 2 * delj[i]):
                        bset.add(p)
                B[i] = bset
        else:
            # 4-approximation
            t = 1 + (1/3)
            for i in range(noc):
                bset = set()
                for p in range(noc):
                    if (p != i and dist[i * noc + p] < t * delj[i]):
                        bset.add(p)
                B[i] = bset

        # Pick the terminal with the smallest delta j and open that as facility
        # Assign any j's whose Bj' overlaps with Bj to facility j
        # Repeat until all the facilities have been assigned
        xint = np.zeros(x.size)
        yint = np.zeros(y.size)
        openedf = set()
        connectedterms = set()
        
        sortdel = delj.argsort()
        for term in sortdel:
            if (term in openedf or term in connectedterms):
                continue
            
            else:
                # Open terminal j since all the connection costs
                # are equal
                yint[term] = 1
                openedf.add(term)
                
                # Connect all terminals j' that intersect with Bj
                for k in range(len(B)):
                    if (k != term and len(B[term] & B[k]) > 0):
                        if (k not in openedf and k not in connectedterms):
                            xint[term * noc + k] = 1
                            connectedterms.add(k)

        return yint, xint, openedf
 
def fetchPlatform():
    print("="*40, "System Information", "="*40)
    uname = platform.uname()
    print(f"System: {uname.system}")
    print(f"Node Name: {uname.node}")
    print(f"Release: {uname.release}")
    print(f"Version: {uname.version}")
    print(f"Machine: {uname.machine}")
    print(f"Processor: {uname.processor}")
    
    return   
 
if __name__ == '__main__':
    flmip = FLLP()
    # Read distances from file
    fname = 'distances.pkl'
    dist = pickle.load(open(fname, 'rb'))
    facility_op_cost = 39.03502850891936
    
    # LP Solver
    start_time = time.monotonic()
    start_mem = psutil.virtual_memory().used
    y, x, fac, con = flmip.solve_lp(dist, f = facility_op_cost, noc = 500)
    print('Facility opening cost (LP):', fac)
    print('Total Connection Cost (LP):', con)
    end_mem = psutil.virtual_memory().used
    end_time = time.monotonic()
    memory_used = end_mem- start_mem
    time_elapsed = timedelta(seconds = end_time - start_time)
    print("Time Elapsed: ", time_elapsed)
    print("Total cores:", psutil.cpu_count(logical=True))
    print('*' * 20)
    
    # Call 6 approximation rounding
    start_time = time.monotonic()
    start_mem = psutil.virtual_memory().used
    fy, fx, facint = flmip.filter_round_solution(y, x, dist)
    print('\nNumber of facilities open', len(facint))
    facop = sum(facility_op_cost * fy)
    iconc = sum(fx * dist)
    print('Facility opening cost (6-approx.):', facop)
    print('Total Connection Cost (6-approx.):', iconc)
    print('Optimal Value of (6-approx.):', facop + iconc)
    end_mem = psutil.virtual_memory().used
    end_time = time.monotonic()
    memory_used = end_mem- start_mem
    time_elapsed = timedelta(seconds = end_time - start_time)
    print("Time Elapsed: ", time_elapsed)
    print("Total cores:", psutil.cpu_count(logical=True))
    print('*' * 20)
    
    fy4, fx4, facint4 = flmip.filter_round_solution(y, x, dist, alpha = True)
    print('\nNumber of facilities open', len(facint4))
    facop4 = sum(facility_op_cost * fy4)
    iconc4 = sum(fx4 * dist)
    print('Facility opening cost (4-approx.):', facop4)
    print('Total Connection Cost (4-approx.):', iconc4)
    print('Optimal Value of (4-approx.):', facop4 + iconc4)
    
    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    print(f"Frequency: {cpufreq.max:.2f}Mhz")
    fetchPlatform()