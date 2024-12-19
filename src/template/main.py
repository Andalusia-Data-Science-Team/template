from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class IrisClassifier:
    """
    A class to classify the Iris dataset using a Decision Tree classifier.

    The class loads the Iris dataset, splits it into training and testing sets,
    trains a Decision Tree model on the training data, makes predictions on the
    test data, and evaluates the model's performance using accuracy.

    Attributes:
    -----------
    iris : Bunch
        The Iris dataset, containing features and labels.
    X : ndarray of shape (n_samples, n_features)
        The feature data (4 features: sepal length, sepal width, petal length, petal width).
    y : ndarray of shape (n_samples,)
        The target labels (0, 1, or 2, corresponding to the species of the iris).
    clf : DecisionTreeClassifier
        The Decision Tree classifier used for model training and prediction.

    Methods:
    --------
    train_model():
        Splits the data into training and test sets, trains the model, and prints
        the accuracy and comparisons between the predicted and actual labels.
    """

    def __init__(self):
        """
        Initializes the IrisClassifier by loading the Iris dataset and setting up the classifier.

        Loads the Iris dataset and stores the feature data (X) and the target labels (y).
        Initializes the DecisionTreeClassifier instance.
        """
        # Load the iris dataset
        self.iris = load_iris()
        self.X = self.iris.data  # Features (4 features for each flower)
        self.y = self.iris.target  # Labels (3 classes of Iris species)

        # Initialize the DecisionTreeClassifier
        self.clf = DecisionTreeClassifier()

    def train_model(self):
        """
        Trains the Decision Tree model on the Iris dataset and evaluates its performance.

        This method performs the following steps:
        1. Splits the dataset into training and test sets using a 80-20 split.
        2. Trains the classifier on the training data.
        3. Makes predictions using the trained model on the test data.
        4. Calculates and prints the accuracy of the model on the test set.
        5. Optionally prints the predicted and actual labels for comparison.

        The accuracy of the model and the comparison between predicted and actual
        labels are printed to the console.
        """
        # Split the data into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Train the classifier on the training data
        self.clf.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = self.clf.predict(X_test)

        # Calculate the accuracy of the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")

        # Optional: Print the predicted and actual labels for comparison
        print(f"Predicted labels: {y_pred}")
        print(f"Actual labels: {y_test}")


# Create an instance of IrisClassifier and train the model
classifier = IrisClassifier()
classifier.train_model()
