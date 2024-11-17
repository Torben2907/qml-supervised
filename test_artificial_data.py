from qmlab.data_generation import generate_random_data

random_state = 42

X_train, y_train, X_test, y_test = generate_random_data(
    feature_dimension=2,
    training_examples_per_class=20,
    test_examples_per_class=10,
    random_state=random_state,
    device="cpu",
)

print(X_train)
print(y_train)
