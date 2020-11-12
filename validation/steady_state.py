def follow_to_steady_state(index, vector, visited_indices):
    """ get to a steady-state, starting at a specific location (index) in the savings path (vector) """

    if index == vector[index]:
        # reached a steady-state
        return index

    if index in visited_indices:
        # in an infinite loop.
        if abs(index - vector[index]) == 1:
            # the loop is between two adjacent elements, so just return their average
            return (index + vector[index])/2

        # the loop is between distant elements. for example, the path might
        # be 0 -> 1, 1 -> 2, 2 -> 0. we report this to the user and halt the
        # script execution. if we'll find that this scenario is likely, we'll
        # find a smarter way to handle it.
        print(vector)
        raise Exception("reached a cycle")

    # not yet in a steady-state, so skip to next element in the path.
    visited_indices.append(index)
    return follow_to_steady_state(vector[index], vector, visited_indices)


def get_steady_states(vector):
    """ returns all steady-states for this savings path (vector) """
    steady_states = []

    # get to a single steady-state from each starting point in the savings path.
    for index, scalar in enumerate(vector):
        steady_states.append(follow_to_steady_state(index, vector, []))

    # return unique list of steady-states
    return set(steady_states)


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
