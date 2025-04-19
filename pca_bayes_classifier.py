import scipy.io
import numpy as np
import matplotlib.pyplot as plt

#Task1
def train_load(trainfile, testfile):
    trains_data = scipy.io.loadmat(trainfile)
    test_data = scipy.io.loadmat(testfile)
    
    x_train01 = trains_data['x']
    y_train01 = trains_data['y'].ravel()     
    x_test01 = test_data['x']
    y_test01 = test_data['y'].ravel()     

    train_samples = x_train01.shape[0]
    test_samples = x_test01.shape[0]
    
    trains_vectorized = x_train01.reshape(train_samples, -1)
    tests_vectorized = x_test01.reshape(test_samples, -1)
    
    return trains_vectorized, tests_vectorized, y_train01, y_test01


trains_data_file = "train_data.mat"
test_data_file = "test_data.mat"

trains_vectorized, tests_vectorized, y_train01, y_test01 = train_load(trains_data_file, test_data_file)

mean = np.mean(trains_vectorized, axis=0)
std = np.std(trains_vectorized, axis=0)


train_data_normalized = (trains_vectorized - mean) / std
test_data_normalized = (tests_vectorized - mean) / std


#Task 2
cov_matrix = np.cov(train_data_normalized,rowvar=False)

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]


#Task 3
train_data_pca = train_data_normalized @ eigenvectors[:, :2]
test_data_pca = test_data_normalized @ eigenvectors[:, :2]

plt.figure(figsize=(10, 8))
plt.scatter(train_data_pca[y_train01 == 0, 0], train_data_pca[y_train01 == 0, 1], 
            label="T-shirt (Train)", alpha=0.5, color='green')
plt.scatter(train_data_pca[y_train01 == 1, 0], train_data_pca[y_train01 == 1, 1], 
            label="Sneaker (Train)", alpha=0.5, color='red')


plt.legend()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection of Training and Testing Data')
plt.show()


#Task4
tshirt_data = train_data_pca[y_train01==0]
sneaker_data = train_data_pca[y_train01==1]

tshirt_mean = np.mean(tshirt_data, axis=0)
tshirt_cov = np.cov(tshirt_data,rowvar=False)

sneaker_mean = np.mean(sneaker_data, axis=0)
sneaker_cov = np.cov(sneaker_data, rowvar=False)


#Task5
prior_tshirt = 0.5
prior_sneaker = 0.5

def classify(sample):
    tshirt_prob1 = multivariate_normal_pdf(sample, tshirt_mean, tshirt_cov)
    sneaker_prob1 = multivariate_normal_pdf(sample, sneaker_mean, sneaker_cov)
    if (tshirt_prob1 > sneaker_prob1):
        return 0
    return 1

def multivariate_normal_pdf(sample, mean, cov):
    k=len(mean)
    diff=sample-mean
    inv_cov=np.linalg.inv(cov)
    exponent = -0.5* np.dot(diff.T, np.dot(inv_cov, diff))
    coeff= 1 / (np.sqrt((2*np.pi) ** k * np.linalg.det(cov)))
    return coeff * np.exp(exponent)
    
train_predictions = np.array([classify(x) for x in train_data_pca])
test_predictions = np.array([classify(x) for x in test_data_pca])

train_accuracy = np.mean(train_predictions == y_train01)
test_accuracy = np.mean(test_predictions == y_test01)

print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
print(f'Testing Accuracy: {test_accuracy * 100:.2f}%')
