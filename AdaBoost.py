import math
import numpy as np
import operator


# NOTE you can use if for Q1 c in tutorial9
# Go to the bottom to add in your own data

# Decision stump used as weak classifier
class DecisionStump():
    def __init__(self, id, classifier=None):
        self.classifier = classifier
        self.id = id
        

    def predict(self, sample):
        # classifier = [operator, x_num, threshold, output]
        if self.classifier[0](sample[self.classifier[1]-1], self.classifier[2]):
            return self.classifier[3]
        else:
            return -self.classifier[3]

    def __repr__(self) -> str:
        return f"{self.id} {self.classifier}"

class Adaboost():

    def __init__(self, classifiers, target_error, max_iterations=10):
        self.n_clf = len(classifiers)
        self.clfs = np.array([])
        for i in classifiers:
            self.clfs = np.concatenate(( self.clfs, [DecisionStump(i, classifiers[i])] ))
        print("Classifiers:\n", self.clfs, '\n')
        # self.alpha = 0
        self.alpha = []
        self.target_error = target_error
        self.max_iterations = max_iterations
        self.best_classifiers = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights to 1/N
        w = np.full(n_samples, (1 / n_samples))

        iteration = 1
        while True:
            print(f"Iteration {iteration}: \nWeights: {w}")

            # Iterate through classifiers and find the best one
            lowest_error = 100000000
            best_classifier = 0
            for i in range(0, self.n_clf):
                clf = self.clfs[i]
                # Calculate Error
                err = 0
                for j, sample in enumerate(X):
                    prediction = clf.predict(sample)
                    error = 0 if prediction == y[j] else 1
                    err += error * w[j]
                if err < lowest_error:
                    lowest_error = err
                    best_classifier = i

            self.best_classifiers.append(self.clfs[best_classifier])
            print(f"Best classifier: {best_classifier+1}")

            # Get predictions from best classifer for each sample
            predictions = []
            for j, sample in enumerate(X):
                prediction = self.clfs[best_classifier].predict(sample)
                predictions.append(prediction)

            # Calculate weighted training error of best classifier
            weighted_error = 0
            for i, prediction in enumerate(predictions):
                # error += 0 if prediction == y[i] else 1
                w_e = 0 if prediction == y[i] else 1
                w_e *= w[i]
                weighted_error += w_e
            error /= len(y)
            print(f"Best classifier's weighted training error: {weighted_error}")
            
            
            # Calculate Alpha
            EPS = 1e-10
            alpha = 0.5 * np.log((1.0 - lowest_error) / (lowest_error))
            self.alpha.append(alpha)
            print(f"Alpha: {alpha}")

            # Calculate weights for next iteration
            new_w = []
            for i, weight in enumerate(w):
                new_w.append(w[i] * (np.exp(- alpha * y[i] * predictions[i])))
                print(f"Update weight: {round(w[i],4)}*e^-{round(alpha,4)}*{y[i]}*{predictions[i]} ----> {round(new_w[-1], 4)}")
            # Normalize to one
            Z_normalisation = 0
            for i, weight in enumerate(new_w):
                Z_normalisation += weight
            for i, weight in enumerate(new_w):
                print(f"Update weight: {round(new_w[i],4)}/{round(Z_normalisation,4)}")
                new_w[i] /= Z_normalisation
            print(f"Normalisation Z{iteration} when updating new weights: {Z_normalisation}")
            # Update weights for next iteration
            w = new_w



            # Check if the classifier has reached the desired target error
            # Find the output*alpha of each classifier for each sample
            tot_error = 0
            decision_formula = ''
            sample_classifications = np.zeros((X.shape[0], len(self.alpha)))
            for i, alpha in enumerate(self.alpha):  
                clf = self.best_classifiers[i]
                for j, sample in enumerate(X):
                    prediction = clf.predict(sample)
                    sample_classifications[j][i] = alpha if prediction == y[j] else -alpha
                decision_formula += f"{alpha} * h{clf.id}(x) + "
            
            # Calculate the AdaBooster classification error in this round
            sample_classifications = sample_classifications.sum(axis=1)
            for i, classification in enumerate(sample_classifications):
                classification = 1 if classification >= 0 else -1
                tot_error += 1/X.shape[0] if classification == y[j] else 0
            print(f"AdaBoost Classifier in this round: {decision_formula[:-2]}")
            print(f"AdaBoost Classifier (unweighted) error in this round: {tot_error}")
            # If the error is below our target error stop the execution
            if tot_error <= self.target_error:
                print('\n')
                print(f"The final hard classifier is: sgn({decision_formula[:-2]})")
                return

            # If we have reached the max iterations stop the execution
            if iteration >= self.max_iterations:
                return
            iteration += 1
            print("\n")



    def predict(self, sample):
        tot_error = 0
        sample_classifications = np.zeros((1, len(self.alpha)))
        for i, alpha in enumerate(self.alpha):  
            clf = self.best_classifiers[i]
            prediction = clf.predict(sample)
            sample_classifications[0][i] = alpha if prediction == 1 else -alpha
        # Calculate the AdaBooster classification error in this round
        sample_classifications = sample_classifications.sum(axis=1)
        return 1 if sample_classifications[0] >= 0 else -1




if __name__ == '__main__':
    
    # Update the below with the dataset you want to use
    dataset = np.array([[1,0], [-1,0], [0,1], [0,-1]])
    labels = np.array([1,1,-1,-1])

    # Enter all the weak classifiers here
    # {classifier_num: [operator, x_num, threshold, output]...}
    classifiers = {1: [operator.gt, 1, -0.5, 1], 2: [operator.gt, 1, -0.5, -1], 3: [operator.gt, 1, 0.5, 1], 4: [operator.gt, 1, 0.5, -1], 5: [operator.gt, 2, -0.5, 1], 6: [operator.gt, 2, -0.5, -1], 7: [operator.gt, 2, 0.5, 1], 8: [operator.gt, 2, 0.5, -1]}
    
    target_error = 0
    max_iterations = 10

    h_err = []

    for h in classifiers:
        err = 0.
        clf = classifiers[h]
        for indx, x in enumerate(dataset):
            if clf[0](x[clf[1]-1], clf[2]):
                out = clf[3]
            else:
                out = -clf[3]
            if labels[indx] != out:
                err += 1.
        h_err.append(err/float(len(dataset)))

    print("Initial classification error: {}\n".format(list(zip(classifiers.keys(), h_err))))
    print("Initial bagging training error: {}\n".format(np.sum(h_err)/float(len(h_err))))

    classifier = Adaboost(classifiers, target_error, max_iterations)
    classifier.fit(dataset, labels)
    
    # Additional sample to classify
    sample = np.array([1,1], dtype=float)
    result = classifier.predict(sample)
    print(f"The AdaBoost classifier classified it as {result}")
