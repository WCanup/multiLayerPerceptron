from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
from mlp import *

auto_mpg = fetch_ucirepo(id=9) 
  
# data (as pandas dataframes) 
X = auto_mpg.data.features 
y = auto_mpg.data.targets 

data = pd.concat([X, y], axis=1)
cleaned_data = data.dropna()

X = cleaned_data.iloc[:, :-1]
y = cleaned_data.iloc[:, -1]

# Display the number of rows removed
rows_removed = len(data) - len(cleaned_data)
print(f"Rows removed: {rows_removed}")

# Do a 70/30 split (e.g., 70% train, 30% other)
X_train, X_leftover, y_train, y_leftover = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,    # for reproducibility
    shuffle=True,       # whether to shuffle the data before splitting
)

# Split the remaining 30% into validation/testing (15%/15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_leftover, y_leftover,
    test_size=0.5,
    random_state=42,
    shuffle=True,
)

# Compute statistics for X (features)
X_mean = X_train.mean(axis=0)  # Mean of each feature
X_std = X_train.std(axis=0)    # Standard deviation of each feature

# Standardize X
X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Compute statistics for y (targets)
y_mean = y_train.mean()  # Mean of target
y_std = y_train.std()    # Standard deviation of target

# Standardize y
y_train = (y_train - y_mean) / y_std
y_val = (y_val - y_mean) / y_std
y_test = (y_test - y_mean) / y_std

X_train.shape

print(f"Samples in Training:   {len(X_train)}")
print(f"Samples in Validation: {len(X_val)}")
print(f"Samples in Testing:    {len(X_test)}")

X_train_np = X_train.to_numpy()
X_val_np = X_val.to_numpy()
X_test_np = X_test.to_numpy()
y_train_np = y_train.to_numpy().reshape(-1, 1)
y_val_np = y_val.to_numpy().reshape(-1, 1)
y_test_np = y_test.to_numpy().reshape(-1, 1)


input_dimensions = X_train_np.shape[1]
y_dimensions = 1

layer_one = Layer(fan_in=input_dimensions, fan_out=32, activation_function=Relu())
layer_two = Layer(fan_in=32, fan_out=32, activation_function=Relu())
layer_three = Layer(fan_in = 32, fan_out=32, activation_function=Relu() )

layer_four = Layer(fan_in =32, fan_out=y_dimensions, activation_function=Linear() )

multiLayerPerceptron = MultilayerPerceptron(layers=(layer_one, layer_two, layer_three,layer_four))

trainingLoss, validationLoss = multiLayerPerceptron.train(train_x=X_train_np, train_y=y_train_np, val_x=X_val_np, val_y=y_val_np, loss_func=SquaredError(), learning_rate=1E-3, batch_size=16, epochs=150)

lossf = SquaredError()

test_predictions = multiLayerPerceptron.forward(X_test_np)
test_loss = lossf.loss(y_true=y_test_np, y_pred=test_predictions)
print(f"\nFinal Test Loss: {test_loss:.4f}")



plt.plot(trainingLoss, label='Training',color='b')
plt.plot(validationLoss, label='Validation', color='r')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve', size = 16)
plt.legend()
plt.show()