import numpy as np
import me570_rrtstar
from tabulate import tabulate as tab
from timeit import default_timer as timer

# Objectives of this project
"""
Comparing convergence rate, memory consumption, and
performance in different environments.
"""

# Convergence Rate
"""
Convergence rate is based on the time elasped. The clock starts
at the same time as the algorithm starts to plan in the environment.
The clock stops when the either the goal is found or ran out of
iterations. 
"""
# Memory Consumption
"""
Memory consumption is based on the number of nodes store within the
tree graph.
"""
# Performances in different environmnets
"""
The performance between the RRT* and RRT*-PNR will be based on both
the convergence rate and memory consumption.
"""
# Important Note
"""
Search radius should be bigger than max step
Search radius should be at least twice compare to max_step.
This is because of the math.log
"""
##### Start testing #####
# initialize grid
xx_grid = [-10,10]
# creating an average storing variable
timer_rrt = 0.0
timer_pnr = 0.0
ratio_rrt = 0.0
ratio_pnr = 0.0
win_rrt = 0
win_pnr = 0
# initialize storing variables
test_pnr = []
test_rrt = []
# initialize planning variables
iter_num = 1000 # number of iterations
iter_planner = 100 # number of testing
search_radius = 6
max_step = 1
env = "sphereworld.mat"
plot_env = True
sample_num = 10 # number of sampling within the rejection probability
# initialize the start and goal node location
if env == 'maze.mat':
    x_start = np.array([[-9],[-9]])
    x_goal = np.array([[4],[8]])
elif env == 'narrowCor.mat':
    x_start = np.array([[6],[9.5]])
    x_goal = np.array([[-8],[-8]])
elif env == 'sphereworld.mat':
    x_start = np.array([[0],[0]])
    x_goal = np.array([[5],[-7]])
elif env == 'clutteredworld.mat':
    x_start = np.array([[-8],[-5]])
    x_goal = np.array([[8],[3]])
# start looping
for _ in range(iter_planner):
    # initialize planning variables
    rrt = me570_rrtstar.RRTStar(x_start=x_start,x_goal=x_goal,
                            search_radius=search_radius,
                            max_step=max_step,
                            iter_num=iter_num,
                            xx_grid=xx_grid,yy_grid=xx_grid,
                            env = env)
    timer,flag_reach,num_nodes=rrt.planning(pnr_active=False,plot_env=plot_env,sample_num=sample_num)
    test_rrt.append((timer,flag_reach,num_nodes))
    if flag_reach is True:
        win_rrt = win_rrt + 1
    pnr = me570_rrtstar.RRTStar(x_start=x_start,x_goal=x_goal,
                            search_radius=search_radius,
                            max_step=max_step,
                            iter_num=iter_num,
                            xx_grid=xx_grid,yy_grid=xx_grid,
                            env = env)
    timer,flag_reach,num_nodes=pnr.planning(pnr_active=True,plot_env=plot_env,sample_num=sample_num)
    if flag_reach is True:
        win_pnr = win_pnr + 1
    test_pnr.append((timer,flag_reach,num_nodes))
# tally the average of both algorithms
for idx in range(iter_planner):
    timer_rrt = timer_rrt + test_rrt[idx][0]
    timer_pnr = timer_pnr + test_pnr[idx][0]
    ratio_rrt = ratio_rrt + test_rrt[idx][2]
    ratio_pnr = ratio_pnr + test_pnr[idx][2]
avg_table = [("RRT*",timer_rrt/iter_planner,ratio_rrt/iter_planner,win_rrt/iter_planner),
             ("RRT*-PNR",timer_pnr/iter_planner,ratio_pnr/iter_planner,win_pnr/iter_planner)]
# creating summary table
print('\n')
print("RRT*-Algorithm:")
print(tab(test_rrt, headers=['Time (s)','Reach?','Tree Nodes']))
print('\n')
print("RRT*-PNR-Algorithm:")
print(tab(test_pnr, headers=['Time (s)','Reach?','Tree Nodes']))
print('\n')
print("Average RRT*-Algorithm Performances:")
print("RRT*-Algorithm V.S. RRT*-PNR-Algorithm:")
print(tab(avg_table, headers=["Algorithm","Avg Time (s)","Avg Tree Nodes","Success Rate"]))
print('\n')