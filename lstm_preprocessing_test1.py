import numpy as np
from sklearn.model_selection import train_test_split
import pickle

# Read in separate input data files and concatenate
print("Reading in data...")
x_bed = np.loadtxt("input_files/xx_1000_60_bed.csv", delimiter=",", dtype=float)
y_bed = np.loadtxt("input_files/yy_1000_60_bed.csv", delimiter=",", dtype=float)
x_bed = np.delete(x_bed, y_bed[:,0]==2, axis=0)
y_bed = np.delete(y_bed, y_bed[:,0]==2, axis=0)

x_fall = np.loadtxt("input_files/xx_1000_60_fall.csv", delimiter=",", dtype=float)
y_fall = np.loadtxt("input_files/yy_1000_60_fall.csv", delimiter=",", dtype=float)
x_fall = np.delete(x_fall, y_fall[:,0]==2, axis=0)
y_fall = np.delete(y_fall, y_fall[:,0]==2, axis=0)

x_pickup = np.loadtxt("input_files/xx_1000_60_pickup.csv", delimiter=",", dtype=float)
y_pickup = np.loadtxt("input_files/yy_1000_60_pickup.csv", delimiter=",", dtype=float)
x_pickup = np.delete(x_pickup, y_pickup[:,0]==2, axis=0)
y_pickup = np.delete(y_pickup, y_pickup[:,0]==2, axis=0)

x_run = np.loadtxt("input_files/xx_1000_60_run.csv", delimiter=",", dtype=float)
y_run = np.loadtxt("input_files/yy_1000_60_run.csv", delimiter=",", dtype=float)
x_run = np.delete(x_run, y_run[:,0]==2, axis=0)
y_run = np.delete(y_run, y_run[:,0]==2, axis=0)

x_sitdown = np.loadtxt("input_files/xx_1000_60_sitdown.csv", delimiter=",", dtype=float)
y_sitdown = np.loadtxt("input_files/yy_1000_60_sitdown.csv", delimiter=",", dtype=float)
x_sitdown = np.delete(x_sitdown, y_sitdown[:,0]==2, axis=0)
y_sitdown = np.delete(y_sitdown, y_sitdown[:,0]==2, axis=0)

x_standup = np.loadtxt("input_files/xx_1000_60_standup.csv", delimiter=",", dtype=float)
y_standup = np.loadtxt("input_files/yy_1000_60_standup.csv", delimiter=",", dtype=float)
x_standup = np.delete(x_standup, y_standup[:,0]==2, axis=0)
y_standup = np.delete(y_standup, y_standup[:,0]==2, axis=0)

x_walk = np.loadtxt("input_files/xx_1000_60_walk.csv", delimiter=",", dtype=float)
y_walk = np.loadtxt("input_files/yy_1000_60_walk.csv", delimiter=",", dtype=float)
x_walk = np.delete(x_walk, y_walk[:,0]==2, axis=0)
y_walk = np.delete(y_walk, y_walk[:,0]==2, axis=0)

x_full = np.concatenate([x_bed, x_fall, x_pickup, x_run, x_sitdown, x_standup, x_walk])
y_full = np.concatenate([y_bed, y_fall, y_pickup, y_run, y_sitdown, y_standup, y_walk])
y_full = y_full[:, 1:]
# x_full = np.concatenate([x_bed, x_fall])
# y_full = np.concatenate([y_bed, y_fall])

# Extract y columns for bed and fall only
# y_full = y_full[:, 0:3]

print(x_full.shape)
print(y_full.shape)

# Split into training and testing data (80/20)
print("Splitting data into training and test sets...")
x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.20, random_state=1000)

# Write out data files
print("Writing out data...")
# np.savetxt("x_train_1.csv", x_train, delimiter=",")
# np.savetxt("x_test_1.csv", x_test, delimiter=",")
# np.savetxt("y_train_1.csv", y_train, delimiter=",")
# np.savetxt("y_test_1.csv", y_test, delimiter=",")
data = [x_train, x_test, y_train, y_test]
with open("data_test3.pk1", "wb") as file:
    pickle.dump(data, file)

print("Done")
