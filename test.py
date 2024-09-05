
from sklearn.preprocessing import OneHotEncoder

categories = [['0', '1', '2']]
data_set = [{"x": ['1', '0'], "y": ['1']}, {"x": ['0', '1'], "y": ['2']}]

for instance in data_set:
    # Change each element x_values so that each is its own array
    x_values = [[element] for element in instance["x"]]

    # Use One Hot Encoding to change these x values
    encoder = OneHotEncoder(categories=categories)
    encoder.fit(x_values)
    one_hot_encoded = encoder.transform(x_values).toarray()
    print("One-hot encoded data: ", one_hot_encoded)

    # Flatten the array to be 1D 
    x_values = [item for subarr in one_hot_encoded for item in subarr]
    print("final x_values: ", x_values)

    # Reassign the x values
    instance["x"] = x_values
    





