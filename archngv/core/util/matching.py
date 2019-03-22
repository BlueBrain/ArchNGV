import numpy as np
from scipy.spatial.distance import cdist

def stable_marriage(women_preferences, men_preferences):
    '''Matches N women to M men so that max(M, N)
    are coupled to their preferred choice that is available
    See https://en.wikipedia.org/wiki/Stable_marriage_problem
    '''
    free_women = list(range(len(women_preferences)))
    free_men = list(range(len(men_preferences)))

    couples = {woman: None for woman in free_women}

    while len(free_women) > 0:

        m = free_men.pop()
        woman_of_choice = men_preferences[m].pop()

        if woman_of_choice in free_women:

            couples[woman_of_choice] = m
            free_women.remove(woman_of_choice)

        else:

            engaged_man = couples[woman_of_choice]

            if women_preferences[woman_of_choice].index(m) > engaged_man:

                free_men.append(engaged_man)
                couples[woman_of_choice] = m

            else:

                free_men.append(m)

    return couples


def find_matching(point_array1, point_array2):

    smallest_group, biggest_group = sorted((point_array1, point_array2), key=len)

    available = np.ones(len(biggest_group), dtype=np.bool)

    distx = cdist(smallest_group, biggest_group)

    s_preference = [np.argsort(row)[::-1].tolist() for row in distx]
    b_preference = [np.argsort(col)[::-1].tolist() for col in distx.T]

    return stable_marriage(s_preference, b_preference)
