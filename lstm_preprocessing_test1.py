import numpy as np
from sklearn.model_selection import train_test_split
import pickle

# Read in separate input data files and concatenate
print("Reading in data...")
x_bed = np.loadtxt("input_files/xx_1000_60_bed.csv", delimiter=",", dtype=float)
y_bed = np.loadtxt("input_files/yy_1000_60_bed.csv", delimiter=",", dtype=float)
x_fall = np.loadtxt("input_files/xx_1000_60_fall.csv", delimiter=",", dtype=float)
y_fall = np.loadtxt("input_files/yy_1000_60_fall.csv", delimiter=",", dtype=float)
x_pickup = np.loadtxt("input_files/xx_1000_60_pickup.csv", delimiter=",", dtype=float)
y_pickup = np.loadtxt("input_files/yy_1000_60_pickup.csv", delimiter=",", dtype=float)
x_run = np.loadtxt("input_files/xx_1000_60_run.csv", delimiter=",", dtype=float)
y_run = np.loadtxt("input_files/yy_1000_60_run.csv", delimiter=",", dtype=float)
x_sitdown = np.loadtxt("input_files/xx_1000_60_sitdown.csv", delimiter=",", dtype=float)
y_sitdown = np.loadtxt("input_files/yy_1000_60_sitdown.csv", delimiter=",", dtype=float)
x_standup = np.loadtxt("input_files/xx_1000_60_standup.csv", delimiter=",", dtype=float)
y_standup = np.loadtxt("input_files/yy_1000_60_standup.csv", delimiter=",", dtype=float)
x_walk = np.loadtxt("input_files/xx_1000_60_walk.csv", delimiter=",", dtype=float)
y_walk = np.loadtxt("input_files/yy_1000_60_walk.csv", delimiter=",", dtype=float)

x_full = np.concatenate([x_bed, x_fall, x_pickup, x_run, x_sitdown, x_standup, x_walk])
y_full = np.concatenate([y_bed, y_fall, y_pickup, y_run, y_sitdown, y_standup, y_walk])
# x_full = np.concatenate([x_bed, x_fall])
# y_full = np.concatenate([y_bed, y_fall])

# Extract y columns for bed and fall only
# y_full = y_full[:, 0:3]

# Convert 2 label for no_activity to 1
y_full[y_full==2] = 1

# Split into training and testing data (80/20)
print("Splitting data into training and test sets...")
x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.20, random_state=0)

# Write out data files
print("Writing out data...")
# np.savetxt("x_train_1.csv", x_train, delimiter=",")
# np.savetxt("x_test_1.csv", x_test, delimiter=",")
# np.savetxt("y_train_1.csv", y_train, delimiter=",")
# np.savetxt("y_test_1.csv", y_test, delimiter=",")
data = [x_train, x_test, y_train, y_test]
with open("data_test1.pk1", "wb") as file:
    pickle.dump(data, file)

print("Done")
