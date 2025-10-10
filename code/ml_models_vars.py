from sklearn.svm import SVC

# SVM Models

model1 = SVC(kernel = 'poly',degree = 5)
model2 = SVC(kernel = 'rbf', C = 450)
model3 = SVC(kernel = 'poly', C = 450, coef0 = 0.5)
model4 = SVC(kernel = 'poly', C = 450, degree = 5)
model5 = SVC(kernel = 'linear', C = 450)