# -*- coding: utf-8 -*-
from functools import reduce

"""
Returns a string with a formatted matrix.

:matrix     Matrix to format.
:offset     Amount of spaces between elements. Defaults to 1.
:padding    Horizontal padding for each element. Defaults to 3.
"""    
def matrix_to_str(matrix, offset=1, padding = 3):
    mstr = ""
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            mstr += str(matrix[i][j]).ljust(padding)
            for k in range(offset):
                mstr += ' '
        mstr += '\n'
    return mstr

"""
Returns a matrix where the values are the indexes indicated by a dictionary.

:idx_dict   Index dictionary.
:matrix     Matrix to index.
"""
def do_indexing(idx_dict,matrix):
    f_col = [row[0] for row in matrix] 
    return (f_col, [[idx_dict[x] for x in row[1:]] for row in matrix])
    
    

def propose(proposer, propose, cap_proposes, pref_proposes, m_proposers, m_proposes):
    """
    Returns True if the proposal is successful and false otherwise.

    :proposer      int
    :propose        int
    :cap_proposes   list(int)
    :pref_proposes  list(int)
    :m_proposers   list(set())
    :m_proposes     list(set())
    """
    success = False
    # If there is place we simply accept
    if(cap_proposes[propose] > len(m_proposes[propose])):
        m_proposes[propose].add(proposer)
        success = True
    else: # If there is no place there are two options
        i = len(pref_proposes)-1 #last element in priority
        cont = True
        while cont:
            if pref_proposes[i] in m_proposes[propose]:
                # 1. Candidate better than worst match -> Accept
                idx_worst = pref_proposes[i]
                m_proposers[idx_worst].remove(propose)
                m_proposes[propose].remove(idx_worst)
                m_proposes[propose].add(proposer)
                success = True
                cont = False
            elif pref_proposes[i] == proposer:
                # 2. Candidate worse than worst match ->  Reject
                cont = False
            else:
                i -= 1
    
    return success
    

def gale_shapley(n_proposers, n_proposes, cap_proposers, cap_proposes, pref_proposers,
                pref_proposes):
    """
    Receives a map of string to int that indicates, for each proposer or proposee
    the index at its preference matrix. Assumes values are indexed in the preference
    matrices.

    :proposers         int
    :proposes           int
    :cap_proposers     list(int)
    :cap_proposes       list(int)
    :pref_proposers    list(list(int))
    :pref_proposes      lsit(list(int)).
    """
    m_proposers = [set() for x in range(n_proposers)]
    m_proposes = [set() for x in range(n_proposes)]
    cont,current = (True,0)
    curr_prop = [0 for x in range(n_proposers)]

    if (n_proposers > 0 and n_proposes > 0):
        while cont:
            while (len(m_proposers[current]) == cap_proposers[current]):
                current = (current+1) % n_proposers
                
            # Everyone must propose once
            i, proposal = curr_prop[current], True # Proposes until accepted
            while (proposal and i < len(pref_proposers[current])):
                proposee_idx = pref_proposers[current][i] 
                
                if(propose(current, proposee_idx, cap_proposes, pref_proposes[proposee_idx],
                        m_proposers, m_proposes)):
                    m_proposers[current].add(proposee_idx) # Update proposer
                    if(len(m_proposers[current]) == cap_proposers[current]):
                        proposal = False # We end the proposal process
                i = i+1
            # At least a proposer is free
            cont = reduce(lambda x,y: (x or len(y[0]) < y[1]), zip(m_proposers,cap_proposers), False)
            # At least a proposee is free
            cont = cont and reduce(lambda x,y: (x or len(y[0]) < y[1]), zip(m_proposes, cap_proposes), False)
    
    return (m_proposers, m_proposes)