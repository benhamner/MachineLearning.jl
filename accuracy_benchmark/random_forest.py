import csv
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import sys

def main():
    output_file = sys.argv[1]
    input_file  = sys.argv[2]
    target_name = sys.argv[3]

    data  = pd.read_csv(input_file)
    train = data[data["is_test"] == False]
    test  = data[data["is_test"] == True]

    x_train = train[[c for c in train.keys() if c != target_name and c != "is_test"]]
    y_train = train[target_name]
    x_test  = test [[c for c in test.keys()  if c != target_name and c != "is_test"]]

    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(x_train, y_train)
    y_hat = rf.predict(x_test)

    f = open(output_file, "w")
    f.write("result\n")
    f.write("\n".join([str(x) for x in y_hat]))
    f.write("\n")
    f.close()

if __name__=="__main__":
    main()