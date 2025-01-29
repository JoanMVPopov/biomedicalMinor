import ast
import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler


#######################################
## This is used for parsing an output.txt file
## In case main.py does not provide the best combos or something else happens
## If the output is the only thing you have, this will give you the leaderboard thing
## Just add END to the final line of the output file

def main():
    cd = os.getcwd()

    options = []
    pairwise_total_sum_list = []
    current_to_first_total_sum_list = []

    with open(os.path.join(cd, "output50.txt"), "r") as file:
        # Iterate through each line in the file
        current_combo = None
        pairwise_total_sum = 0
        current_to_first_total_sum = 0
        diameter_list = []

        for line in file:
            if line.startswith("CURRENT") or line.startswith("END"):
                if current_combo is not None:
                    pairwise_total_sum_list.append(pairwise_total_sum)
                    current_to_first_total_sum_list.append(current_to_first_total_sum)

                    options.append(
                        {
                            'pairwise_total_sum': pairwise_total_sum,
                            'current_to_first_total_sum': current_to_first_total_sum,
                            'params': current_combo
                        }
                    )

                    pairwise_total_sum = 0
                    current_to_first_total_sum = 0
                    diameter_list = []

                if line.startswith("CURRENT"):
                    current_combo = ast.literal_eval(line.split("PARAMS: ")[1])
                    print(current_combo)

                continue

            elif line.startswith("RUN"):
                diameter = np.float64(line.split("DIAMETER: ")[1])

                if len(diameter_list) > 0:
                    diameter_penalty = 0
                    pairwise_abs_diff_to_append = (abs(diameter - diameter_list[-1])) ** 2
                    current_to_first_abs_diff_to_append = (abs(diameter - diameter_list[0])) ** 2

                    if diameter < 100:
                        # diameter_penalty = abs((diameter/100) - 1)
                        diameter_penalty = abs(diameter - 100) ** 3
                    elif diameter > 150:
                        # diameter_penalty = abs((diameter/150) - 1)
                        diameter_penalty = abs(diameter - 100) ** 3

                    # pairwise_abs_diff.append(pairwise_abs_diff_to_append + (diameter_penalty * pairwise_abs_diff_to_append)**2)
                    # current_to_first_abs_diff.append(current_to_first_abs_diff_to_append + (diameter_penalty * current_to_first_abs_diff_to_append)**2)

                    pairwise_total_sum += (-1)*(pairwise_abs_diff_to_append + diameter_penalty)
                    current_to_first_total_sum += (-1)*(current_to_first_abs_diff_to_append + diameter_penalty)

                diameter_list.append(diameter)
            else:
                continue

    print(len(options))

    numpy_array_for_minmax_analysis = np.column_stack((pairwise_total_sum_list, current_to_first_total_sum_list))

    scaler = MinMaxScaler()

    # apply MinMaxScaler to each column
    scaled_data = scaler.fit_transform(numpy_array_for_minmax_analysis)  # Scale each column to [0, 1]
    # multiply by 100 to spread out results even more
    scaled_data_100 = scaled_data * 100

    # sum the results for each row
    # it's essentially like a leaderboard (places decided by sum of the two features per each row)
    # this way we can take both results into account
    # e.g.:
    # f1 f2
    # 100 25 -> 125 (2nd place)
    # 50 60 -> 110 (3rd place)
    # 70 80 -> 150 (1st place)
    row_sums = scaled_data_100.sum(axis=1)

    for i, row in enumerate(row_sums):
        options[i]['final_score'] = row

    sorted_options = sorted(options, key=lambda x: x['final_score'], reverse=True)

    for i in range(10):
        print(f"CHOICE {i}\nSCORE: {sorted_options[i]['final_score']}\t"
              f"F1: {sorted_options[i]['pairwise_total_sum']}\t F2: {sorted_options[i]['current_to_first_total_sum']}"
              f"\nPARAMS: {sorted_options[i]['params']}\n")



if __name__ == "__main__":
    main()