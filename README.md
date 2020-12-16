An implementation of an instance of a k-median and facility location problem.

500 of the US's largest cities' longitude and latitude are inputted via 500_us_city_coords.txt. A 1-swap local search algorithm and its facility location approximation is implemented in:
localSearch.ipynb.

A 4 and 6 LP approximation is located in:
flmip.py

Phase 1 of the primal dual solution is in:
dps.ipynb 
while phase 2 of the primal dual solution is in:
dps.py
The dual primal solutions have not been integrated, but are very nearly implemented. 
