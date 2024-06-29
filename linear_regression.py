import random

# Generate some random data
def generate_data(num_points, slope, intercept, noise):
    x = [random.uniform(0, 10) for _ in range(num_points)]
    y = [slope * xi + intercept + random.uniform(-noise, noise) for xi in x]
    return x, y

# Implement linear regression
def linear_regression(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_xx = sum(xi * xi for xi in x)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    intercept = (sum_y - slope * sum_x) / n
    
    return slope, intercept

# Make predictions
def predict(x, slope, intercept):
    return [slope * xi + intercept for xi in x]

# Calculate mean squared error
def mean_squared_error(y_true, y_pred):
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)

# Main program
if __name__ == "__main__":
    # Generate synthetic data
    true_slope, true_intercept = 2, 1
    x_train, y_train = generate_data(100, true_slope, true_intercept, noise=0.5)
    
    # Train the model
    learned_slope, learned_intercept = linear_regression(x_train, y_train)
    
    # Make predictions
    y_pred = predict(x_train, learned_slope, learned_intercept)
    
    # Calculate error
    mse = mean_squared_error(y_train, y_pred)
    
    # Print results
    print(f"True slope: {true_slope}, True intercept: {true_intercept}")
    print(f"Learned slope: {learned_slope:.4f}, Learned intercept: {learned_intercept:.4f}")
    print(f"Mean squared error: {mse:.4f}")