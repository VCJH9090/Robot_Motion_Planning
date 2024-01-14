import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy import io as scio
from matplotlib.lines import Line2D
from timeit import default_timer as timer

class Node:
    """
    Node class for Tree Graph
    """
    def __init__(self,vertex):
        self.vertex = vertex
        # if isinstance(vertex, np.ndarray):
        #     print("my_array is a numpy array")
        # else:
        #     print("my_array is not a numpy array")
        self.parent = None

class RRTStar:
    """
    RRT* Algorithm Paper:
    https://journals.sagepub.com/doi/epdf/10.1177/0278364911406761
    """
    def __init__(self,x_start,x_goal,search_radius,max_step,iter_num,xx_grid,yy_grid,env):
        # initialize the start and goal node
        self.start = Node(x_start)
        self.goal = Node(x_goal)
        # tree graph
        self.vertices = np.array(x_start) # initialize starting vertex
        self.nodes = np.array([self.start]) # initialize starting node
        self.edges = [] # initialize empty edge set
        # initialize the iteration number
        self.iter_num = iter_num
        # initialize max step variable
        self.max_step = max_step
        # initialize search radius
        self.search_radius = search_radius
        # initialize a distance and theta variable
        self.dist = 0.0
        self.theta = 0.0
        # grid space xx_grid and yy_grid (2x1 matrix)
        self.xx_grid = xx_grid # [lower_bound, upper_bound]
        self.yy_grid = yy_grid # [lower_bound, upper_bound]
        # initialize obstacle class
        self.obs = Obstacles(self.xx_grid,self.yy_grid,env)
        # initialize flag variable
        self.flag_reach = False
        self.exclude = np.array([]) # keeps all the excluded node after collision check
        self.rejected = np.array([]) # keeps all the rejected node
        
    def planning(self,pnr_active,plot_env,sample_num):
        """
        The algorithm starts planning the path from the
        start to the goal location.
        """
        # begin the timer
        start_time = timer() # seconds
        # collision test for start and goal node
        if self.is_occluded_circle(self.start.vertex):
            return print("Node_start occluded in circle")
        if self.is_occluded_rec(self.start.vertex):
            return print("Node_start occluded in rectangle")
        if self.is_occluded_circle(self.goal.vertex):
            return print("Node_goal occluded in circle")
        if self.is_occluded_rec(self.goal.vertex):
            return print("Node_goal occluded in rectange")
        # if no collision, continue with planning
        for _ in range(self.iter_num):
            # generate random node in the grid space
            n_rand = self.sample_free()
            # find the nearest node in the tree graph
            n_nearest = self.nearest(n_rand)
            # steer the nearest node to the random node
            n_new = self.steer(n_nearest)
            # Rejection Probability
            if pnr_active: # if true
                n_new = self.rejection(n_nearest,n_new,sample_num)
            # check for obstacle-free
            if not self.not_obstacle_free(n_nearest.vertex,n_new.vertex):
                # obtain the nearest vertices idx within a radius
                near_idx = self.near(n_new)
                # include n_new into the tree vertices
                self.vertices = np.hstack((self.vertices,n_new.vertex))
                self.nodes = np.append(self.nodes,n_new)
                if len(near_idx) > 0:
                    # establishing min_path_cost and add edges
                    self.min_cost_path(near_idx,n_new,n_nearest)
                    # rewire the tree 
                    self.rewire(near_idx,n_new)
            else:
                self.exclude = np.append(self.exclude,n_new)
            # check if the current trees reaches close to the goal
            if np.linalg.norm(n_new.vertex - self.goal.vertex) <= self.max_step:
                # check whether there is a collision free path
                if not self.not_collision_free(n_new.vertex,self.goal.vertex):
                    self.flag_reach = True
                    break # break out of the loop
        # check whether the tree actually reach near the goal
        if self.flag_reach:
            end_timer = timer() # seconds
            # print("Goal Reach within a radius step of", self.max_step)
        else: # not reach
            end_timer = timer() # seconds
            # print("Not enough iterations to reach the goal")
        if plot_env: # plotting the environment and resulting path
            # extract path
            path = self.extract_path(self.nodes[-1])
            # animate first then plot the final path
            self.animation(path, abs(end_timer-start_time), self.flag_reach)
        return abs(end_timer-start_time),self.flag_reach,len(self.nodes)
    
    def rejection(self,n_nearest,n_new,n_times):
        """
        Node rejection probability:
        
        Activated if the new node, after the steer function, detected within the region of
        distance influence. Then use its nearest node to sample n times within its radius.
        
        First, reject nodes outside of the region of distance influnce with zero repulsive
        function. Each remaining node will be selected to compute for repulsive function.
        Probability for each node will be computed and selected for new node that has the
        lowest probability, where it is less likely to collide with the obstacle.
        """
        if not self.not_collision_free(n_nearest.vertex,n_new.vertex):
            if len(self.obs.obs_rect_list) > 0: # rectangle obstacles
                # obtain list of detection within the distance of influence with the new node
                rect_dist = self.rect_dist_inf(n_new.vertex)
                if rect_dist: # new node detected within region
                    return self.reject_rect_node(n_nearest,n_new,n_times)
                else: # the new node is not within the distance of influence
                    return n_new
            elif len(self.obs.obs_cir_list) > 0: # circle obstacles
                # obtain list of detection within the distance of influence with the new node
                cir_dist = self.cir_dist_inf(n_new.vertex)
                if cir_dist: # new node detected within region
                    return self.reject_cir_node(n_nearest,n_new,n_times)
                else: # new node is not within the distance of influence
                    return n_new
            elif len(self.obs.obs_bound_list) > 0: # line obstacles
                # obtain list of detection within the distance of influence with the new node
                line_dist = self.line_dist_inf(n_new.vertex)
                if line_dist: # new node detected within region
                    return self.reject_line_node(n_nearest,n_new,n_times)
                else: # new node is not within the dsitance of influence
                    return n_new
        else:
            return n_new
    
    def reject_cir_node(self,n_nearest,n_new,n_times):
        """
        Rejects all nodes that are either not within the distance of
        influence or have higher probability of getting collided.
        """
        # initialize the list of Repulsive Function Computation
        u_rep = 0.0 # to store the repulsion value
        rep_cir = {} # to store the key and value for each node
        rep_cir_prob = {} # to store the probability for each node
        # sample for n times from the n_nearest node
        cir_xy = self.rand_rsample(n_times,n_nearest.vertex)
        cir_xy = np.hstack([cir_xy,n_new.vertex])
        # account only for nodes within the distance of influence
        for idx in range(len(cir_xy)):
            # obtain list of detections
            inf_dict = self.cir_dist_inf(cir_xy[:,[idx]])
            # chek whether there are detections
            if not(inf_dict): # no detections
                continue
            else: # detected within the region of influence
                # (key => dist: value => idx_cir)
                for dict_idx,dict_dist in enumerate(inf_dict):
                    if dict_idx == 0: # initialize the first repulsion value
                        u_rep = self.rep_potential(dict_dist,
                            self.obs.obs_cir_list[inf_dict[dict_dist]].dist_inf)
                    else:
                        if len(inf_dict) > 1: # more than two obstacles detected
                            temp_rep = self.rep_potential(dict_dist,
                                self.obs.obs_cir_list[inf_dict[dict_dist]].dist_inf)
                            if temp_rep > u_rep: # select the max repulsion value
                                u_rep = temp_rep
                # store the {key => repulsive: val => idx}
                rep_cir[u_rep] = idx
        # check whether there is a list of repulsion values
        if not(rep_cir):
            return n_new # return the current n_new
        # compute the overall sum of repulsive potential value
        sum_rep = sum(rep_cir)
        # compute the Rejection Probability
        for key,val in rep_cir.items():
            # store the {key => proability: val => idx}
            rep_cir_prob[key/sum_rep] = val
        # find the lowest repulsion probability value
        min_rep = min(rep_cir_prob)
        # get the newest new node from the detections list
        n_newest = cir_xy[:,[rep_cir_prob[min_rep]]]
        if not(np.array_equal(n_newest,n_new.vertex)):
            self.rejected = np.append(self.rejected,n_new)
        return Node(n_newest)
        
    def reject_rect_node(self,n_nearest,n_new,n_times):
        """
        Rejects all nodes that are either not within the distance of
        influence or have higher probability of getting collided.
        """
        # initialize the list of Repulsive Function Computation
        u_rep = 0.0 # to tally the repulsion value for each node
        # u_dist = 0.0 # to tally the distance to each node
        rep_rect = {} # to store the key and value for each node
        rep_rect_prob = {} # to store the probability for each node
        # sample for n times from the n_nearest node
        rec_xy = self.rand_rsample(n_times,n_nearest.vertex)
        rec_xy = np.hstack([rec_xy,n_new.vertex])
        # account only for nodes within the distance of influence
        for idx in range(len(rec_xy)):
            # obtain list of detections
            inf_dict = self.rect_dist_inf(rec_xy[:,[idx]])
            # check whether there are detections
            if not(inf_dict): # no detections
                continue
            else: # detected within the region of influence
                # (key => dist: value => (idx_vertex,rect_idx))
                for dict_idx,dict_dist in enumerate(inf_dict):
                    if dict_idx == 0: # initialize the first repulsive value
                        u_rep = self.rep_potential(dict_dist,
                            self.obs.obs_rect_list[inf_dict[dict_dist][1]].dist_inf)
                    else:
                        if len(inf_dict) > 1: # more than two obstacles detected
                            temp_rep = self.rep_potential(dict_dist,
                            self.obs.obs_rect_list[inf_dict[dict_dist][1]].dist_inf)
                            if temp_rep > u_rep: # select the max repulsion value
                                u_rep = temp_rep
                        # # compare if the second repulsion is smaller
                        # temp_rep = self.rep_potential(dict_dist,
                        #     self.obs.obs_rect_list[inf_dict[dict_dist][1]].dist_inf)
                        # if len(inf_dict) < 2: # only one obstacle detected
                        #     print("one obstacle")
                        #     if temp_rep < u_rep: # select the min repulsion value
                        #         u_rep = temp_rep # replace the current repulsion value
                        # else: # the node is within more than 1 obstacle
                        #     print(two obstacle)
                        #     if temp_rep > u_rep: # select the max repulsion value
                        #         u_rep = temp_rep
                # for key,val in inf_dict.items():
                #     # compute repulsive function
                #     u_rep = u_rep + self.rep_potential(val[1],
                #             self.obs.obs_rect_list[key].dist_inf)
                #     u_dist = u_dist + val[1]
                    
                # scale the u_rep according to the total distance to each obstacle
                # u_rep = u_rep / u_dist
                # store the {key => repulsive: val => idx}
                rep_rect[u_rep] = idx # avg repulsive value
        # check whether there is a list of repulsion values
        if not(rep_rect):
            return n_new # return the current n_new
        # compute the overall sum of repulsive potential value
        sum_rep = sum(rep_rect)
        # compute Rejection Probability
        for key,val in rep_rect.items():
            # store the {key => probability: val => idx}
            rep_rect_prob[key/sum_rep] = val
        # find the lowest repulsion probability value
        min_rep = min(rep_rect_prob)
        # get the newest new node from the detections list
        n_newest = rec_xy[:,[rep_rect_prob[min_rep]]]
        if not(np.array_equal(n_newest,n_new.vertex)):
            self.rejected = np.append(self.rejected,n_new)
        return Node(n_newest)
    
    def reject_line_node(self,n_nearest,n_new,n_times):
        """
        Rejects all nodes that are either not within the distance of
        influence or have higher probability of getting collided.
        """
        # initialize the list of Repulsive Function Computation
        u_rep = 0.0 # to tally the repulsion value for each node
        rep_line = {} # to store the key and value for each node
        rep_line_prob = {} # to store the probability for each node
        # sample for n times from the n_nearest node
        line_xy = self.rand_rsample(n_times,n_nearest.vertex)
        line_xy = np.hstack([line_xy,n_new.vertex])
        # account only for nodes within the distance of influence
        for idx in range(len(line_xy)):
            # obtain list of detections
            inf_dict = self.line_dist_inf(line_xy[:,[idx]])
            # check whether there are detections
            if not(inf_dict): # no detections
                continue
            else: # detected within the region of influence
                # (key => dist: value => (idx_vertex,rect_idx))
                for dict_idx,dict_dist in enumerate(inf_dict):
                    if dict_idx == 0: # initialize the first repulsive value
                        u_rep = self.rep_potential(dict_dist,
                            self.obs.obs_bound_list[inf_dict[dict_dist][1]].dist_inf)
                    else:
                        if len(inf_dict) > 1: # more than two obstacles detected
                            temp_rep = self.rep_potential(dict_dist,
                            self.obs.obs_bound_list[inf_dict[dict_dist][1]].dist_inf)
                            if temp_rep > u_rep: # select the max repulsion value
                                u_rep = temp_rep
                # store the {key => repulsive: val => idx}
                rep_line[u_rep] = idx # avg repulsive value
        # check whether there is a list of repulsion values
        if not(rep_line):
            return n_new # return the current n_new
        # compute the overall sum of repulsive potential value
        sum_rep = sum(rep_line)
        # compute Rejection Probability
        for key,val in rep_line.items():
            # store the {key => probability: val => idx}
            rep_line_prob[key/sum_rep] = val
        # find the lowest repulsion probability value
        min_rep = min(rep_line_prob)
        # get the newest new node from the detections list
        n_newest = line_xy[:,[rep_line_prob[min_rep]]]
        if not(np.array_equal(n_newest,n_new.vertex)):
            self.rejected = np.append(self.rejected,n_new)
        return Node(n_newest)
        
    def rep_potential(self,dist,dist_inf):
        """
        Computing the repulsive potential value given a node
        within the region of influence
        """
        # compute the repulsive potential function
        u_rep = 0.5*((1/dist)-(1/dist_inf))**2
        return u_rep
    
    def rand_rsample(self,n_times,n_nearest):
        """
        https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly
        
        Once the new node is detected within the distance of influence region.
        Begin sampling randomly witin the search radius.
        
        Note that simply picking an angle [0,2pi] and radius [0,1] using the
        random function is not random. To maintain uniform density within the
        circle with a defined radius, the number of sampling is scale depends
        on the defined radius r, having a probability density function to be
        linear starting from 0 to 2pi. To acheive random sampling, it is better
        to sample randomly in a parallelogram and project back into the circle,
        where sampling for (origin,x,y,r) => ABCD of parallelogram. 
        """
        # initialize the sampling radius based on current number of vertices
        v = len(self.vertices)+1
        sample_radius = min(self.search_radius*math.sqrt((math.log(v)/v)),self.max_step)
        # sample_radius = self.max_step
        # defining the angle
        theta = 2*math.pi*np.random.uniform(0.0,sample_radius,n_times) % (2*math.pi) # sample R=>[0,sample_radius]
        # theta = np.array(theta)
        # defining the parallelogram
        r = np.random.uniform(0.0,sample_radius,n_times) + np.random.uniform(0.0,sample_radius,n_times)
        r = np.array(r)
        # project the random points back to the circle
        r = r % sample_radius
        # plot the sample radius
        # ax = plt.gca()
        # kwargs = {'fill':False}
        # ax.add_patch(
        #     plt.Circle(xy=(n_nearest[0,0],n_nearest[1,0]),
        #                radius=sample_radius,
        #                edgecolor='c',
        #                **kwargs)
        # )
        # compute for coordinates in cartesian space
        cart_cord = np.array([n_nearest[0,0] + (r*np.cos(theta)),
                              n_nearest[1,0] + (r*np.sin(theta))])    
        return cart_cord  
    
    def rect_dist_inf(self, n_new):
        """
        Returns a dictionary of a detected distance_influence
        """
        # initialize dictionary (key => dist: value => (idx_vertex,rect_idx))
        rect_dist = {}
        # check whether the new node is within the region of dist_influence
        for idx,rect in enumerate(self.obs.obs_rect_list): # Rectangle
            # compute the n_new distance to each rectangle
            vertices = np.hstack([rect.vertices,rect.vertices[:,[0]]])
            for v_idx in range(len(vertices[0])-1):
                # get the shortest distance
                dist,_ = getshortdist(vertices[:,[v_idx]],vertices[:,[v_idx+1]],n_new)
                # the closest distance from the node would be bounded between the parametric line
                if (0 < dist < rect.dist_inf):
                    rect_dist[dist] = (v_idx,idx) # within the dist_inf
                # if 0.0 < param < 1.0: # node bounded within the parametric line
                #     if (0 < dist < rect.dist_inf) and \
                #     not self.is_occluded_rec(n_new): 
                #         rect_dist[idx] = (v_idx,dist) # within the dist_inf
                # elif param < 0: # node near the first vertex
                #     if (dist < )
                    
                # if not(0.0 < param < 1.0):
                #     continue
                # else:   
                #     if (0 < dist < rect.dist_inf) and \
                #     not self.is_occluded_rec(n_new): 
                #         rect_dist[idx] = (v_idx,dist) # within the dist_inf
        return rect_dist
    
    def line_dist_inf(self, n_new):
        """
        Returns a dictionary of a detected distance_influence
        """
        # initialize dictionary (key => dist: value => (idx_vertex,rect_idx))
        line_dist = {}
        # check whether the new node is within the region of dist_influence
        for idx,line in enumerate(self.obs.obs_bound_list): # Line obstacles
            if idx == 0: # grid boundary
                vertices = np.hstack([line.vertices,line.vertices[:,[0]]])
            else: # lines
                vertices = line.vertices
            for v_idx in range(len(vertices[0])-1):
                # get the shortest distance
                dist,_ = getshortdist(vertices[:,[v_idx]],vertices[:,[v_idx+1]],n_new)
                # the closest distance from the node would be bounded between the parametric line
                if (0 < dist < line.dist_inf):
                    line_dist[dist] = (v_idx,idx) # within the dist_inf
        return line_dist
        
    def cir_dist_inf(self, n_new):
        """
        Returns a dictionary of a detected distance_influence
        """
        # initialize dictionary (key => dist : value => cir_index)
        cir_dict = {}
        # check whether there is a list of circles
        if self.obs.obs_cir_list and not self.is_occluded_circle(n_new):
            # check whether the new node is within the region of dist_influence
            for idx,cir in enumerate(self.obs.obs_cir_list): # Circle
                # compute the n_new distance to each circle
                if cir.radius > 0: # Filled
                    dist = np.linalg.norm(n_new - cir.center) - cir.radius
                else: # Hollow
                    dist = np.linalg.norm(n_new - cir.center) + abs(cir.radius)
                if 0 < dist < cir.dist_inf: # within dist_inf
                    cir_dict[dist] = idx
        return cir_dict
                
    def animation(self,path, timer, flag_reach):
        """
        Animate the path
        """
        # create figure
        _ = plt.figure()
        # plotting the environment
        self.obs.grid_plot()
        self.obs.plot_cir('black')
        self.obs.plot_rec('red')
        self.obs.plot_boundary('black')
        # enable plotting interation mode
        plt.ion()
        for idx,node in enumerate(self.nodes):
            if idx == 0: # plotting the starting node
                plt.plot(node.vertex[0,0],node.vertex[1,0],'*r')
                plt.plot(self.goal.vertex[0,0],self.goal.vertex[1,0],'*g')
                continue
            # plot the next node
            plt.plot(node.vertex[0,0],node.vertex[1,0],'*b')
            plt.quiver(node.parent.vertex[0,0],
                       node.parent.vertex[1,0],
                       node.vertex[0,0]-node.parent.vertex[0,0],
                       node.vertex[1,0]-node.parent.vertex[1,0],
                       color='g',
                       angles='xy',
                       scale_units='xy',
                       scale=1.0,
                       width=0.005)
            plt.pause(0.1)
        plt.ioff() # off the interactive mode
        # plot the final path
        self.final_plot(path, timer, flag_reach)
        plt.show()
    
    def final_plot(self,path,timer,flag_reach):
        """
        Plotting the final path
        """
        # reverse the path order to start-to-goal
        path = path[:,::-1]
        # plot the final path
        for node in self.exclude:
            plt.plot(node.vertex[0,0],node.vertex[1,0],'m*')
        for node in self.rejected:
            plt.plot(node.vertex[0,0],node.vertex[1,0],'c*')
        plt.plot(path[0],path[1],'--r',linewidth=2)
        if flag_reach is True: # planner reach the goal location
            plt.title("Elasped Time:{time} s || Number of Tree Node: {num_nodes}"
                .format(time=timer,num_nodes=len(self.nodes)),fontsize = 8)
        else:
            plt.title("Unsearchable, not enough samplings/iterations", fontsize = 8)
        handles = [Line2D([0],[0],marker='*',lw=0,color='m',label='Collided Nodes'),
                   Line2D([0],[0],marker='*',lw=0,color='c',label='Rejected Nodes'),
                   Line2D([0],[0],marker='*',lw=0,color='b',label='Tree Nodes'),
                   Line2D([0],[0],marker='*',lw=0,color='r',label='Start Node'),
                   Line2D([0],[0],marker='*',lw=0,color='g',label='Goal Node')]
        plt.legend(handles=handles,loc='upper left',fontsize=8)
        plt.xlabel("x-coordinates",fontsize=8)
        plt.ylabel("y-coordinates",fontsize=8)
        plt.gca().set_aspect("equal")

    def extract_path(self, n_end):
        """
        Extract path from start to goal
        """
        path = np.array(self.goal.vertex)
        holder = n_end
        
        while holder.parent is not None:
            path = np.hstack((path,holder.vertex))
            holder = holder.parent
        path = np.hstack((path,holder.vertex))
        return path
  
    def sample_free(self):
        """
        Random node sample needs to be independent and identically distributed (i.i.d).
        Additionally, the generated random node has not been checked for collision-free.
        """
        # generate new nodes based on grid space
        # xx_grid and yy_grid have [0,10]
        # random generated node will be bounded between ([0,10],[0,10])
        return Node(np.array([[np.random.uniform(self.xx_grid[0],self.xx_grid[1])],
                     [np.random.uniform(self.yy_grid[0],self.yy_grid[1])]]))
        
    def nearest(self,n_rand):
        """
        Given the tree graph and the random generated node:
        Find the nearest vertex to the random node.
        """
        nearest_idx = self.nearest_neighbor(n_rand.vertex)
        n_nearest = self.nodes[nearest_idx]
        return n_nearest
    
    def nearest_neighbor(self,n_rand):
        """
        Compute the distance to find the nearest neighbor.
        Then find the nearest neighbor index.
        """
        # compute the distance and angle
        vertices = (n_rand - self.vertices)**2
        angles = n_rand - self.vertices
        vertices = np.sum(vertices,axis=0)
        vertices = np.sqrt(vertices)
        # find the smallest distance
        nearest_idx = np.argmin(vertices)
        # re-initialize the dist var and theta var
        self.dist = vertices[nearest_idx]
        self.theta = np.arctan2(angles[1][nearest_idx],angles[0][nearest_idx])
        return nearest_idx
    
    def steer(self,n_nearest):
        """
        Connect the random node to its nearest neighbor. The steering
        mechanism used here is Euclidean Distance (Straight Line)
        
        Here is where we implement our own modified RRT* with
        REJECTION NODE PROBABILITY BASED ON REPULSIVE FUNCTION
        """
        # check if the random node is quite far from the nearest neighbor
        dist = min(self.max_step,self.dist) # choose the smallest distance
        # initialize a new node
        return Node(np.array([[n_nearest.vertex[0,0]+dist*np.cos(self.theta)],
                              [n_nearest.vertex[1,0]+dist*np.sin(self.theta)]]))
    
    def near(self,n_new):
        """
        Find vertices that are contained in a ball of radius r 
        center at n_new. The radius of the ball is size according to
        the cardinality of vertices V (number of vertices in the tree).
        More vertices means smaller ball radius for searching nearby nodes.
        
        Return the nearest vertices as their index.
        """
        # initialize the number of vertices in the tree graph
        v = len(self.vertices)+1 # cardinality of v
        # finding the minimum search radius
        r = min(self.search_radius*math.sqrt((math.log(v))/v), self.max_step)
        # finding the nearest vertices based on n_new node
        near_vertex = np.linalg.norm(n_new.vertex - self.vertices,axis=0)
        near_idx = [idx for idx in range(len(near_vertex)) if near_vertex[idx] <= r+0.01]
        return near_idx # return the vertex index near the n_new
    
    def min_cost_path(self, near_idx, n_new, n_nearest):
        """
        Connect along a minimum-cost path.
        """
        # compute the current minimum cost based Euclidean Distance
        c_min = self.cost(n_nearest) + self.heuristic(n_nearest.vertex,n_new.vertex)
        n_min = n_nearest # initialize the min cost node
        # connecting and finding the min cost path
        for idx in near_idx:
            c_cost = self.cost(self.nodes[idx]) + self.heuristic(self.nodes[idx].vertex,n_new.vertex)
            if not self.not_collision_free(self.nodes[idx].vertex,n_new.vertex) and c_cost <= c_min:
                # re-initialize minimum near node
                n_min = self.nodes[idx]
                # re-initialize min cost
                c_min = c_cost
                # re-initialize parent of n_new
                n_new.parent = n_min
        # append edge into the tree's edge
        self.edges.append((n_min,n_new))
    
    def rewire(self,near_idx,n_new):
        """
        rewire
        """
        for idx in near_idx:
            if not self.not_collision_free(self.vertices[:,[idx]],n_new.vertex):
                if self.cost(n_new) + self.heuristic(self.vertices[:,[idx]],n_new.vertex) < self.cost(self.nodes[idx]):
                    self.nodes[idx].parent = n_new
    
    def heuristic(self,node,n_new):
        """
        Compute the distance between the n_new and its nearest neighbor
        """
        return np.linalg.norm(n_new-node)
    
    def cost(self,node):
        """
        Using Euclidean Distance as the cost notion from current to starting node
        """
        # initializeing cost and holder
        holder = node
        cost = 0.0
        # compute the total cost from start to current node
        while holder.parent is not None:
            cost = cost + np.linalg.norm(holder.vertex-holder.parent.vertex)
            holder = holder.parent
        return cost

    def not_collision_free(self,node,n_new):
        """
        Checking for collision between the node and n_new
        """
        if self.is_intersect_rec(node,n_new) and len(self.obs.obs_rect_list) > 0:
            return True
        if self.is_intersect_circle(node,n_new) and len(self.obs.obs_cir_list) > 0:
            return True
        if self.is_intersect_line(node,n_new) and len(self.obs.obs_bound_list) > 0:
            return True
        return False
    
    def not_obstacle_free(self,n_nearest,n_new):
        """
        Check for collision between the line_n_new_n_nearest and
        its surrounding obstacles (rectangles and circles)
        """
        # check collisions against rectangles
        if len(self.obs.obs_rect_list) > 0:
            if self.is_occluded_rec(n_new):
                return True
            if self.is_intersect_rec(n_nearest,n_new):
                return True
        # check collision against circles
        if len(self.obs.obs_cir_list) > 0:
            if self.is_occluded_circle(n_new):
                return True
            if self.is_intersect_circle(n_nearest,n_new):
                return True
        # check collision against lines
        if len(self.obs.obs_bound_list) > 0:
            # if self.is_occluded_line(n_new):
            #     return True
            if self.is_intersect_line(n_nearest,n_new):
                return True
        return False # obstacle free
    
    def is_occluded_circle(self,node):
        """
        Check whether the node is occluded in the circle obstacles
        Note that there are hollow(-radius) and filled(+radius) obstacles
        """
        # begin checking
        for cir in self.obs.obs_cir_list:
            # compute the distance between center to node
            dist_center_node = np.linalg.norm(node - cir.center)
            # compute the distance between center to circle edge
            dist_center_radius = abs(cir.radius)
            # check whether it is hollow or filled
            if cir.radius < 0: # hollow circle
                if dist_center_node > dist_center_radius:
                    return True # node occluded
            else: # filled circle
                if dist_center_node < dist_center_radius:
                    return True # node occluded
        return False # no occlusion of node

    def is_occluded_line(self,node):
        """
        Check whether the ndoe is occluded within the line obstacles.
        """
        # setting tolerance
        tol = 1e-06
        # begin checking
        flag = False
        for idx,line in enumerate(self.obs.obs_bound_list):
            if idx == 0: # grid boundary
                # initialize vertices
                vertices = np.hstack([line.vertices,line.vertices[:,[0]]])
            else: # lines
                # initialize vetices
                vertices = line.vertices
            # check for occlusion using paramteric curve line
            for ind in range(len(vertices)-1):
                t1_x = vertices[0,ind+1] - vertices[0,ind]
                t1_y = vertices[1,ind+1] - vertices[1,ind]
                # check for horizontal or vertical line
                if -tol < t1_x < tol: # vertical line
                    t1 = (node[1,0] - vertices[1,ind]) / t1_y
                elif -tol < t1_y < tol: # horizontal line
                    t1 = (node[0,0] - vertices[0,ind]) / t1_x
            # check whether the node is occluded in the line
            if tol < t1 < 1.0 - tol:
                flag = True # node occlusion detected
        return flag
    def is_occluded_rec(self,node):
        """
        Check whether the node is occluded in the rectangle obstacles
        Note that there are hollow (CW) and filled (CCW) obstacles
        """
        # begin checking
        flag = False
        for rect in self.obs.obs_rect_list:
            if rect.type is True or rect.type == 1: # Filled
                # check for x and y bound
                if rect.vertices[0,0] < node[0,0] < rect.vertices[0,1] and \
                    rect.vertices[1,1] < node[1,0] < rect.vertices[1,2]:
                    flag = True
            if rect.type is False or rect.type == 0: # Hollow
                # check for x and y bound
                if not (rect.vertices[0,0] < node[0,0] < rect.vertices[0,1] and \
                    rect.vertices[1,1] < node[1,0] < rect.vertices[1,2]):
                    flag = True
        return flag
                
    def is_intersect_rec(self,n_nearest,n_new):
        """
        Check for line intersection with rectangle obstacles.
        Line between the new node and its nearest neighbors
        """
        # setting tolerance
        tol = 1e-06
        # initialize edge
        node_edge = np.hstack([n_nearest,n_new])
        for rect in self.obs.obs_rect_list:
            # initialize vertices
            vertices = np.hstack([rect.vertices,rect.vertices[:,[0]]])
            # extract individual edge from each rectangle
            for idx in range(len(vertices[0])-1):
                # establish A_matrix
                # a_mat = np.hstack([vertices[:,[idx+1]]-vertices[:,[idx]],
                #                    -(node_edge[:,[1]]-node_edge[:,[0]])])
                a_mat = np.array([[vertices[0,idx+1].item() - vertices[0,idx].item(),
                                   -(node_edge[0,1] - node_edge[0,0])],
                                  [vertices[1,idx+1].item() - vertices[1,idx].item(),
                                   -(node_edge[1,1] - node_edge[1,0])]],dtype='float32')
                # establish b_vector
                # b_vec = node_edge[:,[0]]-vertices[:,[idx]]
                b_vec = np.array([[node_edge[0,0] - vertices[0,idx].item()],
                                  [node_edge[1,0] - vertices[1,idx].item()]],dtype='float32')
                # check for parallel lines
                if abs(np.linalg.det(a_mat)) < tol:
                    continue
                # solve the linear system
                t_one,t_two = np.linalg.solve(a_mat,b_vec)
                # check that collision point is strictly between endpoints of each edge
                if (tol < t_one.item() < 1.0-tol) and (tol < t_two.item() < 1.0-tol):
                    return True # edge collision detected               
        return False # no edge collision
    
    def is_intersect_line(self,n_nearest,n_new):
        """
        Check for line intersection with line obstacles.
        """
        # setting tolerance
        tol = 1e-06
        # initialize edge
        node_edge = np.hstack([n_nearest,n_new])
        for line in self.obs.obs_bound_list:
            if len(line.vertices[0]) > 2: # grid boundary
                vertices = np.hstack([line.vertices,line.vertices[:,[0]]])
            else: # lines
                vertices = line.vertices
            for idx in range(len(vertices[0])-1):
                a_mat = np.array([[vertices[0,idx+1].item() - vertices[0,idx].item(), 
                                   -(node_edge[0,1] - node_edge[0,0])],
                                  [vertices[1,idx+1].item() - vertices[1,idx].item(),
                                   -(node_edge[1,1] - node_edge[1,0])]],dtype='float32')
                b_vec = np.array([[node_edge[0,0] - vertices[0,idx].item()],
                                  [node_edge[1,0] - vertices[1,idx].item()]],dtype='float32')
                # check for parallel lines
                if abs(np.linalg.det(a_mat)) < tol:
                    continue
                # solve the linear system
                t_one,t_two = np.linalg.solve(a_mat,b_vec)
                # check that collision point is strictly between endpoints of each edge
                if (tol < t_one.item() < 1.0-tol) and (tol < t_two.item() < 1.0-tol):
                    # print("yes")
                    return True # edge collision detected 
        return False # no edge collision 
                
    def is_intersect_circle(self,n_nearest,n_new):
        """
        Check for line intersection with circle obstacles.
        Line between the new node and its nearest neighbors
        """
        for cir in self.obs.obs_cir_list:
            dist,param = getshortdist(n_nearest,n_new,cir.center)
            if dist < cir.radius:
                if 0 < param < 1:
                    return True
        return False
            
class Obstacles:
    """
    Contains list for both rectangular and circular obstacles.
    Store rectangles and circles as objects/structures in the list.
    """
    def __init__(self,xx_grid,yy_grid,env):
        self.obs_rect_list = []
        self.obs_cir_list = []
        self.obs_bound_list = []
        self.xx_grid = xx_grid # list/array [lower_bound, upper_bound]
        self.yy_grid = yy_grid # list/array [lower_bound, upper_bound]
        self.env = env # select the environment
        if self.env == 'narrowCor.mat':
        # import rectangles
            data = scio.loadmat('FinalProject/narrowCor.mat')
            for rect_idx in range(len(data['world'][0])):
                self.add_rect(data['world'][0,rect_idx][0], # vertices
                            data['world'][0,rect_idx][2].item(),# type 
                            data['world'][0,rect_idx][1].item()) # dist_inf
        elif self.env == 'maze.mat':
            # import bound lines
            data = scio.loadmat('FinalProject/maze.mat')
            for line_idx in range(len(data['world'][0])):
                self.add_bound(data['world'][0,line_idx][0],
                            data['world'][0,line_idx][1].item())
        elif self.env == 'sphereworld.mat':
            # import spherical obstacles
            data = scio.loadmat('FinalProject/sphereworld.mat')
            for sphere_args in np.reshape(data['world'],(-1, )):
                self.add_cir(sphere_args[1].item(),sphere_args[0],sphere_args[2].item())
        elif self.env == 'clutteredworld.mat':
            # import spherical obstacles
            data = scio.loadmat('FinalProject/clutteredworld.mat')
            for sphere_args in np.reshape(data['world'],(-1, )):
                self.add_cir(sphere_args[1].item(),sphere_args[0],sphere_args[2].item())
        else:
            print("Please select the following options for the environment:\n")
            print("1) narrowCor.mat\n")
            print("2) maze.mat\n") 
            print("3) sphereworld.mat\n") 
            print("4) clutteredworld.mat\n")     
        
    def add_rect(self,vertices,hollow_filled,dist_influence):
        """
        Adds rectangle obstacles with four vertices, starting with (xy)
           +--------------------------+
           |                          |
         height    Filled-Rectangle   |
           |                          |
          (xy)--------width-----------+
        """
        rect = Rectangle(vertices,hollow_filled,dist_influence)
        self.obs_rect_list.append(rect)
    
    def add_cir(self,radius,center,dist_influence):
        """
        Adds circle obstacles
        """
        cir = Circle(radius,center,dist_influence)
        self.obs_cir_list.append(cir)
        
    def add_bound(self,vertices,dist_influence):
        """
        Adds boundary lines for maze-like environment.
        """
        bound = Boundary(vertices,dist_influence)
        self.obs_bound_list.append(bound)
        
    def grid_plot(self):
        """
        Plot the outline of the environment
        """
        ax = plt.gca()
        ax.set_xlim([self.xx_grid[0], self.xx_grid[1]])
        ax.set_ylim([self.yy_grid[0], self.yy_grid[1]])
        ax.axis('equal')
        
    def plot_cir(self,color):
        """
        Plot the circle obstacles
        """
        # get current axes
        ax = plt.gca()
        # add circle as a patch
        for cir in self.obs_cir_list:
            if cir.radius > 0: # circle is filled in
                kwargs = {'facecolor': (0.3,0.3,0.3)}
                rad_inf = cir.radius + cir.dist_inf
            else: # circle is hollow
                kwargs = {'fill': False}
                rad_inf = -cir.radius - cir.dist_inf
            # initialize center
            center = (cir.center[0,0],cir.center[1,0])
            # adds cirle into the environment
            ax.add_patch(
                plt.Circle(center,
                           radius=abs(cir.radius),
                           edgecolor=color,
                           **kwargs)
            )
            # adds its distance influence of the circle
            ax.add_patch(
                plt.Circle(center,
                           radius=rad_inf,
                           edgecolor=(0.7,0.7,0.7),
                           fill=False)
            )
    
    def plot_rec(self,color):
        """
        Plot the rectangle obstacles
        
           +--------------------------+
           |                          |
         height    Filled-Rectangle   |
           |                          |
          (xy)--------width-----------+
    
        """
        # get current axes
        ax = plt.gca()
        # add rectangles as patch
        for rec in self.obs_rect_list:
            # compute the width and height of filled rectangle
            width = np.linalg.norm(rec.vertices[:,[1]]-rec.vertices[:,[0]])
            height = np.linalg.norm(rec.vertices[:,[-1]]-rec.vertices[:,[0]])
            # check whether the rectangle is hollow or filled
            if rec.type is False or rec.type == 0: # hollow rectangle
                kwargs = {'fill': False}
                dist_inf = rec.dist_inf * -1
            else: # filled rectangle
                kwargs = {'facecolor': (0.3,0.3,0.3)}
                dist_inf = rec.dist_inf
            
            # adds rectangles into the environment
            ax.add_patch(
                plt.Rectangle(xy=(rec.vertices[0,0],rec.vertices[1,0]),
                              width=width,
                              height=height,
                              edgecolor=color,
                              **kwargs)
            )
            # adds its distance influence of the rectangle
            # ax.add_patch(
            #     patches.FancyBboxPatch(xy=(rec.vertices[0,0]-dist_inf,rec.vertices[1,0]-dist_inf),
            #                         width=width+2*dist_inf,
            #                         height=height+2*dist_inf,
            #                         edgecolor=(0.7,0.7,0.7),
            #                         boxstyle = 'Round4, pad=0.01',
            #                         fill=False)
            # )
            ax.add_patch(
                patches.Rectangle(xy=(rec.vertices[0,0]-dist_inf,rec.vertices[1,0]-dist_inf),
                                    width=width+2*dist_inf,
                                    height=height+2*dist_inf,
                                    edgecolor=(0.7,0.7,0.7),
                                    joinstyle='round',
                                    fill=False)
            )
        
    def plot_boundary(self,color):
        """
        Plot boundaries for maze obstacles.
        
           +--------------------------+
           |                          |
         height    Filled-Rectangle   |
           |                          |
          (xy)--------width-----------+
          
        """
        # get current axes
        ax = plt.gca()
        # add boundary lines
        for idx,line in enumerate(self.obs_bound_list):
            kwargs = {'fill': False}
            if idx == 0: # grid boundary
                # compute the width and height
                width = np.linalg.norm(line.vertices[:,[1]] - line.vertices[:,[2]])
                height = np.linalg.norm(line.vertices[:,[3]] - line.vertices[:,[2]])
                vertices = np.hstack([line.vertices,line.vertices[:,[0]]])
            else: # line boundaries
                # compute the differences in lines
                diff = line.vertices[:,[1]] - line.vertices[:,[0]]
                if diff[0,0] != 0: # vertical line
                    width = 0.0
                    height = diff[1,0]
                else: # horizontal line
                    width = diff[0,0]
                    height = 0.0
                vertices = line.vertices
            # adds lines
            plt.plot(vertices[0,:],vertices[1,:],color='black')
            # adds its distance of influence
            if width > 0 and height > 0: # grid boundary
                ax.add_patch(
                    plt.Rectangle(xy=(vertices[0,2]+line.dist_inf,
                                      vertices[1,2]+line.dist_inf),
                                  width=width-2*line.dist_inf,
                                  height=height-2*line.dist_inf,
                                  edgecolor=color,
                                  **kwargs)
                )  
            if width == 0.0 and diff[1,0] > 0.0: # positive height
                ax.add_patch(
                    plt.Rectangle(xy=(vertices[0,0]-line.dist_inf,
                                      vertices[1,0]-line.dist_inf),
                                  width=width+2*line.dist_inf,
                                  height=diff[1,0]+2*line.dist_inf,
                                  edgecolor=(0.7,0.7,0.7),
                                  **kwargs)
                )
            # if width == 0.0 and diff[1,0] < 0.0: # negative height
            #     ax.add_patch(
            #             plt.Rectangle(xy=(line.vertices[0,0]-line.dist_inf,
            #                             line.vertices[1,0]+line.dist_inf),
            #                         width=width+2*line.dist_inf,
            #                         height=height-2*line.dist_inf,
            #                         edgecolor=color)
            #     )
            if height == 0.0 and diff[0,0] > 0.0: # positive width
                ax.add_patch(
                        plt.Rectangle(xy=(line.vertices[0,0]-line.dist_inf,
                                        line.vertices[1,0]-line.dist_inf),
                                    width=diff[0,0]+2*line.dist_inf,
                                    height=height+2*line.dist_inf,
                                    edgecolor=(0.7,0.7,0.7),
                                    **kwargs)
                )
            # if height == 0.0 and diff[0,0] < 0.0: # negative width
            #     ax.add_patch(
            #             plt.Rectangle(xy=(line.vertices[0,0]+line.dist_inf,
            #                             line.vertices[1,0]-line.dist_inf),
            #                         width=width-2*line.dist_inf,
            #                         height=height+2*line.dist_inf,
            #                         edgecolor=color)
            #     )
            
class Rectangle:
    """
    Create rectangle obstacles in the grid space.
    Array size of (2 x nb_vertices) per rectangle
    Four corners ====> Four Vertex.
    
    Hollow => False
    Filled => True
    """
    def __init__(self,vertices,hollow_filled,dist_influence):
        # contains both filled or hollow rectangles
        self.vertices = vertices
        self.type = hollow_filled
        self.dist_inf = dist_influence
            
class Circle:
    """
    Create circle obstacles in the grid space.
    Array size of (2 x nb_circles)
    """
    def __init__(self,radius,center,dist_influence):
        # contains both filled or hollow circles
        self.radius = radius
        self.center = center
        self.dist_inf = dist_influence

class Boundary:
    """
    Create the walls of a maze for instance.
    Array size of (2 x nb_vertices)
    """
    def __init__(self,vertices,dist_influence):
        self.vertices = vertices
        self.dist_inf = dist_influence

def getshortdist(n1,n2,p):
    """
    https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    """
    tol = 1e-06
    a = float(p[0,0])-float(n1[0,0])
    b = float(p[1,0])-float(n1[1,0])
    c = float(n2[0,0])-float(n1[0,0])
    d = float(n2[1,0])-float(n1[1,0])
    
    len_sq = c * c + d * d
    if not(-tol < len_sq < tol):
        dot = a * c + b * d
        param = dot / len_sq
    else:
        param = -1.0
        
    if param < 0:
        xx = n1[0,0]
        yy = n1[1,0]
    elif param > 1:
        xx = n2[0,0]
        yy = n2[1,0]
    else:
        xx = n1[0,0] + param * c
        yy = n1[1,0] + param * d
    
    dx = p[0,0] - xx
    dy = p[1,0] - yy
    
    return math.sqrt(dx*dx + dy*dy), param

def angle(vertex0, vertex1, vertex2, angle_type='unsigned'):
    """
    Compute the angle between two edges  vertex0-- vertex1 and  vertex0--
    vertex2 having an endpoint in common. The angle is computed by starting
    from the edge  vertex0-- vertex1, and then ``walking'' in a
    counterclockwise manner until the edge  vertex0-- vertex2 is found.
    """
    # tolerance to check for coincident points
    tol = 2.22e-16

    # compute vectors corresponding to the two edges, and normalize
    vec1 = vertex1 - vertex0
    vec2 = vertex2 - vertex0

    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 < tol or norm_vec2 < tol:
        # vertex1 or vertex2 coincides with vertex0, abort
        edge_angle = math.nan
        return edge_angle

    vec1 = vec1 / norm_vec1
    vec2 = vec2 / norm_vec2

    # Transform vec1 and vec2 into flat 3-D vectors,
    # so that they can be used with np.inner and np.cross
    vec1flat = np.vstack([vec1, 0]).flatten()
    vec2flat = np.vstack([vec2, 0]).flatten()

    c_angle = np.inner(vec1flat, vec2flat)
    s_angle = np.inner(np.array([0, 0, 1]), np.cross(vec1flat, vec2flat))

    edge_angle = math.atan2(s_angle, c_angle)

    angle_type = angle_type.lower()
    if angle_type == 'signed':
        # nothing to do
        pass
    elif angle_type == 'unsigned':
        edge_angle = (edge_angle + 2 * math.pi) % (2 * math.pi)
    else:
        raise ValueError('Invalid argument angle_type')

    return edge_angle

        
# def euclidean_distance(self,vertices,):
#     """
#     Compute the 
#     """
#     # replicate a copy of the vertices list
#     vertices = self.vertices.copy()
#     # compute the distance 
#     vertices = (n_rand - vertices)**2
#     vertices = np.sum(vertices,axis=0)
#     vertices = np.sqrt(vertices)
#     return vertices