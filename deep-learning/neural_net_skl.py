from sklearn.neural_network import MLPClassifier
# create the model
model = MLPClassifier()

# train the model
model.fit(X,y)

# predict using the trained model
model.predict(X)

