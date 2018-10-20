from sklearn.datasets import fetch_lfw_people
import numpy as np

people = fetch_lfw_people()
counts = np.bincount(people.target)

for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print(f"{name:30} {count:3}", end='    ')
    if (i+1) % 3 == 0:
        print()
