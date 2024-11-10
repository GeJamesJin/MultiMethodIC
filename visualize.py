import matplotlib.pyplot as plt

mp.load

num_features = X.shape[1]  # numeber of feature
num_cols = 4  # adjust later
num_rows = (num_features + num_cols - 1) // num_cols

fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))
axs = axs.flatten()

# Plot each feature

for i in range (0,2):
  for j in range (0,4):
    axs[i, j].scatter(X_1[:,i*4+j+1],y, s=1)
    axs[i, j].set_xlabel(feature_names[i*4+j])
    axs[i, j].set_ylabel('Price')



fig.tight_layout()
plt.show()