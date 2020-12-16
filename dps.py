#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pickle

class DPS:
    def dual_primal_phase1(self, noc, dist, f):
        nocsq = noc**2
        alpha = np.zeros(noc)
        beta = np.zeros(nocsq)
        tol = 0.0001
        
        # Raise the alphas of each city uniformly at each time step t
        t = 0
        while(True):
            alpha += 1.0
            
            for idx in range(alpha.size):
                ai = alpha[idx]
                dijs = [x for x in range(idx, nocsq, noc)]
                tight = np.where(np.abs(ai - dist[dijs]) <= tol)[0]
                # There is an index issue that I need to fix!!
                # Indices for tight won't match up with dist[dijs]
                if (tight.size > 0):
                    i = tight[0]
                    while (ai - beta[i * noc + idx] <= dist[i * noc + idx]):
                        beta[i * noc + idx] += 1.0
            
            t += 1
        
        return
    
    def dual_primal_phase2(self, T, w):
        tprime = set()
        
        while (len(T) > 0):
            i = np.random.choice(T)
            tprime.add(i)
            
            for h in T:
                for j in range(500):
                    if (w[i][j] > 0 and w[h][j] > 0):
                        T.remove(h)        
        return tprime
    
if __name__ == '__main__':
    dp = DPS()
    noc = 500
    
    dist = pickle.load(open('distances.pkl', 'rb'))
    facility_op_cost = 39
    
    dp.dual_primal_phase1(noc, dist, f = facility_op_cost)