#!/usr/bin/env python3

import random
import pandas as pd
import ant_colony_system as acs
import matplotlib.pyplot as plt
import time

pingInfoFilePath = "./costmap.ftr"
cost_db = pd.read_feather(pingInfoFilePath, columns=None, use_threads=True)

# Number of connections:
# (n_cluster - 1) * n_nodes_per_cluster^2 * n_cluster

nodes = list(cost_db.root.unique())
ik_number = {}
for n in nodes:
    ik_number[n] = int(1 + cost_db.loc[cost_db["root"] == n]["root_ik_number"].max())

solver_param = {"beta": 2,
                "rho": 0.1,
                "ph0": 1/(12*len(nodes)),
                "q0": 0.9,
                "dph": 1/(12*len(nodes)),
                "ph_bounds": tuple([ x/(12*len(nodes)) for x in [0.1, 1.05]]),
                "max_no_improv": float("inf"),
                }

# Add Pheromone column
cost_db.insert(cost_db.shape[1], "ph", solver_param["ph0"] * cost_db.shape[0])

# Add Heuristic column
cost_db.insert(cost_db.shape[1], "pweight", [1.0] * cost_db.shape[0])
acs.update_pweights(cost_db, beta=solver_param["beta"])

TESTS = 10
temp = obtained_at = best_ant = [0 for x in range(TESTS)]

for idx in range(TESTS):
    print(f"Iteration: {idx}")
    tic = time.perf_counter()
    best_ant[idx], obtained_at[idx], _ = acs.ant_colony_optimization(10, 100, cost_db, nodes, ik_number, solver_param)
    toc = time.perf_counter()
    temp[idx] = toc - tic
    cost_db["ph"] = solver_param["ph0"]
    cost_db["pweight"] = 1.0
    acs.update_pweights(cost_db, beta=solver_param["beta"])

print(f"Tempi: {temp}")

best_cost = [x["cost"] for x in best_ant]

fig, ax = plt.subplots()

ax.scatter(obtained_at, best_cost)
ax.grid(True)

ax.set_xlabel("Obtained at x(th) iteration")
ax.set_ylabel("Cost")

plt.show()
