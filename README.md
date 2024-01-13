# Path Planning using RRT* with Repulsive Potential for Probabilistic Node Rejection
## Motivation
The objective for this project is to incoporate a notion of obstacle avoidance into the RRT* algorithm to enhance the robustness in the path planning of the algorihtm. To maintain relatively the same convergence rate and increase the likelihood of reducing the memory consumption of the RRT* algorithm, a probabilistic node rejection (PNR) function is embedded into the algorithm that uses the repulsive potential function from the Artificial Potential Field (APF) method. Hence, our algorithm RRT*-PNR consist of both the usage of RRT* and APF methods to obtain a strong collision-free path, resulting in a more robust path planning algorithm while maintaining the probabilistic completeness and asymtotically optimal properties with low level control complexity.
## Robotic System & Scenarios
The robot used within this project is a point robot moving in four different environments which are the normal (sphereworld), maze, narrow corridor, and cluttered environments.
### Performance Metrics/Criterion
Our RRT*-PNR algorithm is evaluated against the RRT* algorithm using the following performance metrics which are convergence rate (average time (s)), memory consumption (average tree nodes), and robustness in the planned path in four different environments.
## Running the main_results.py file
### Testing variables used
To compare both algorithms fairly, the following variables in the main_results.py are kept constant through the testing:
- Number of Iterations (iter_num)                         = 1000
- Number of Testing (iter_planner)                        = 100
- Searching Radius (search_radius)                        = 6
- Maximum Step (max_step)                                 = 1
- Number of Sampling within the PNR Function (sample_num) = 10

The changing variables in the main_results.py file is shown below:
- Choosing Environments (env)      = "sphereworld.mat"
                                   = "maze.mat"
                                   = "narrowCor.mat"
                                   = clutteredworld.mat
- Plotting Environments (plot_env) = True
                                   = False
### Python Dependencies (Python Required Packages)
- Numpy:
    - To store values in arrays and do matrices computations
- Matplotlib:
    - To plot environments with interactive animations
- Math:
    - To compute trigonometries, logarithms, and angles
- Scipy:
    - To import environments' parameters from the .mat file
- Timeit:
    - To track and compute the time differences before and after the sampling iterations (Convergence Rate)
- Tabulate:
    - To print the summary of the results into the terminal for visualization purpose.
### How to run the code used in testing?
To run the main_results.py file, the zip file should contain another python file called "me570_rrtstar.py". Within that file, it stores all the utilities and functions needed to obtain the results and its following plotted environments. To run the code efficiently, do read the subsection "Testing variable used". After changing the testing variables, all it is left to do is to run the main_results.py file.
### How to run the code in general?
All the variables mentioned in the subsection "Testing variable used" can be changed to test with different parameters in different selected environments. The important thing to note is that the search radius must be at least twice bigger in value than the maximum step size because of the usage of logarithms within the variable sampling radius function.



