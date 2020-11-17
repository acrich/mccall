from numba import njit, int64
import numpy as np
from numba.typed import List


@njit
def follow_to_steady_state(index, vector, visited_indices):
    """ get to a steady-state, starting at a specific location (index) in the savings path (vector) """

    if index == vector[index]:
        # reached a steady-state
        return index

    if index in visited_indices:
        # in an infinite loop.
        # the loop is between two or more elements. for example, the path might
        # be 0 -> 1, 1 -> 2, 2 -> 0. we return one of the elements in between,
        # including the edges, and don't worry about which of the elements
        # it is that's returned, since this is likely simply a matter of
        # the grid being too coarse.
        return round(np.mean(np.asarray([index, vector[index]])))

    # not yet in a steady-state, so skip to next element in the path.
    visited_indices.append(index)
    return follow_to_steady_state(vector[index], vector, visited_indices)


def get_steady_states(vector):
    """ returns all steady-states for this savings path (vector) """
    # using the hack from: https://github.com/numba/numba/issues/2152#issuecomment-254520288
    # also using typed lists, as per: https://numba.pydata.org/numba-doc/latest/reference/deprecation.html#deprecation-of-reflection-for-list-and-set-types
    steady_states = List()
    steady_states.append(1)

    # get to a single steady-state from each starting point in the savings path.
    for index, scalar in enumerate(vector):
        visited_indices = List()
        # see comment above about typed lists, etc.
        # len(vector) is an index that isn't ever in the vector, so it doesn't affect the result, only the list's typing.
        visited_indices.append(len(vector))
        steady_states.append(follow_to_steady_state(index, vector, visited_indices))

    # return unique list of steady-states
    return set(steady_states[1:])


def get_steady_state(vector):
    """ returns the lowest steady-state in this savings path (vector) """

    steady_states = get_steady_states(vector)

    if len(steady_states) == 0:
        return None

    if len(steady_states) > 1:
        print(steady_states)

    # next(iter(set)) returns the first element in the set.
    # by sorting first, we ensure that we get the lowest steady-state.
    return next(iter(sorted(steady_states)))
