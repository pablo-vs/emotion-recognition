from math import sqrt
import numpy as np
import sys
import itertools

from matching import *

# Tracking function (naïve version)
def gen_identity(current_n, base='Agent '):
    return base + str(current_n)

def gen_identities(n_agents, base='Agent ', offset=0):
    """
    Generates string identities for agents.
    :n_agents int
    :base str
    :return list(str)
    """
    return [gen_identity(i+offset) for i in range(n_agents)]

def metric(B1, B2):
    """ Given two bounding boxes, returns the value of the metric. """
    return np.linalg.norm(B1[0:2] - B2[0:2]) + np.log(1+abs(B1[2]-B2[2])) + np.log(1+abs(B1[3]-B2[3]))

def naive_track(assig, expected, centers_t, tolerance, tol=2, A=0):
    """
    Tracks the given agents: associates each point 
    of centers_t with an agent identity given the previous
    assignment, the expected detections, and the current tolerance
    value for each agent.

    The return values are a map from identity to point, which stores
    the last known position of each agent; and a list
    of strings, where the string in index i is the identity of the
    ith point in centers_t.

    :assig      dict(str->point)
    :expected   dict(str->point)
    :centers_t  list(point)
    :tolerance  dict(str->int)
    :tol        int
    :A          int
    :return     (dict(str->point),list(str))
    """
    # Heavily affected by granularity in time discretization   
    n,m = len(expected), len(centers_t)
    previous_centers = list(expected.items()) #[(agent,val) for agent,val in expected.items()]
    # Create "distance" matrix (len(expected) x len(centers_t))
    d_matrix = np.array([[metric(bb_prev,bb_det) for bb_det in centers_t] for _,bb_prev in expected.items()])   
    prefs_prev = np.array([sorted(list(range(m)), key=lambda j:d_matrix[i][j]) for i in range(n)]) # sort by distance (rows)
    prefs_det = np.array([sorted(list(range(n)), key=lambda i:d_matrix[i][j]) for j in range(m)]) # sort by distance (columns)
    # Call GS algorithm (returns list of size 1 sets)
    match_prev, match_det = gale_shapley(n, m, [1 for _ in range(n)], [1 for _ in range(m)], prefs_prev, prefs_det)
    match_prev = [next(iter(x)) if len(x) != 0 else None for x in match_prev]
    match_det = [next(iter(x)) if len(x) != 0 else None for x in match_det]
    # Tracking will work properly if the agents' range of movement is smaller than
    # 1/(2*tol) times their separation
    assignment_dict = dict()
    assignment_list = []

    for i in range(m):
        if match_det[i] is None: # New agents
            identity = gen_identity(A)
            assignment_dict[identity] = centers_t[i]
            assignment_list.append(identity)
            A += 1 # upper bound for last assigned agent number
        else:   # Existing agent
            identity = previous_centers[match_det[i]][0]
            assignment_dict[identity] = centers_t[i]
            assignment_list.append(identity)
        tolerance[identity] = tol

    for i in range(n):
        if match_prev[i] is None: # Misdetection or agent left
            identity = previous_centers[i][0]
            if tolerance[identity] > 0: # Add to match with last position
                assignment_dict[identity] = assig[identity]
                assignment_list.append(identity) # These will appear after all detections
                tolerance[identity] -= 1
            else:
                del tolerance[identity]
                print("Deleting agent",identity)
                print("Current assignment is", assignment_dict)
    
    return assignment_dict, assignment_list, A
    

def transform_bbox(x,y,w,h,nw,nh,ow,oh):
    """
    Transform an nw*nx bounding box into an ow*oh one.

    :return (int,int,int,int)
    """
    #x / nw = xp / ow
    xp = int(ow * x / nw)
    yp = int(oh * y / nh)
    wp = int(ow * w / nw)
    hp = int(oh * h / nh)
    return (xp,yp,wp,hp)
