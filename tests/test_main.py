import pytest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from template.main import IrisClassifier

# Test Class: Test the behavior of IrisClassifier


class TestIrisClassifier:
    """
    Test suite for the IrisClassifier class.
    This suite will test the following functionalities:
    1. Class initialization
    2. Model training and prediction
    3. Accuracy score evaluation
    """

    @pytest.fixture
    def classifier(self):
        """
        Fixture to initialize the IrisClassifier class.
        This fixture provides a fresh instance of the classifier
        for each test.
        """
        return IrisClassifier()

    def test_classifier_initialization(self, classifier):
        """
        Test the initialization of the IrisClassifier class.
        It checks if the dataset is loaded properly and if the classifier
        is correctly initialized.
        """
        # Check if the Iris dataset is loaded
        assert classifier.X is not None
        assert classifier.y is not None
        assert len(classifier.X) == 150  # 150 samples in the Iris dataset
        assert len(classifier.y) == 150

        # Check if the classifier is properly initialized
        assert isinstance(classifier.clf, DecisionTreeClassifier)

    def test_train_model(self, classifier):
        """
        Test the model training process.
        This checks if the classifier can train on the Iris dataset.
        """
        classifier.train_model()  # Training the model

        # Since the train_model method prints the accuracy, we are not testing for return value,
        # but we can check if the model training process runs without exceptions.

        # We will assume the training process is correct if it completes without error.

    def test_accuracy_with_train_model(self, classifier):
        """
        Test if the classifier gives an accuracy score that is above a reasonable threshold.
        This tests that the model is working and provides a non-trivial accuracy score.
        """
        # Split the dataset for testing
        X_train, X_test, y_train, y_test = train_test_split(classifier.X, classifier.y, test_size=0.2, random_state=42)
        classifier.clf.fit(X_train, y_train)

        # Make predictions
        y_pred = classifier.clf.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Check if the accuracy is above a reasonable threshold (e.g., 70%)
        assert accuracy > 0.7

    def test_predict_method(self, classifier):
        """
        Test the prediction method after training.
        This checks if the model can predict on unseen data.
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(classifier.X, classifier.y, test_size=0.2, random_state=42)

        # Train the classifier
        classifier.clf.fit(X_train, y_train)

        # Make predictions
        y_pred = classifier.clf.predict(X_test)

        # Check if predictions are made
        assert y_pred is not None
        assert len(y_pred) == len(y_test)  # Number of predictions should match number of test samples

    def test_model_not_overfitting(self, classifier):
        """
        Test if the classifier is not overfitting.
        We check if accuracy on the training data is similar to the accuracy on test data.
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(classifier.X, classifier.y, test_size=0.2, random_state=42)

        # Train the classifier
        classifier.clf.fit(X_train, y_train)

        # Evaluate the classifier on both training and test data
        train_accuracy = classifier.clf.score(X_train, y_train)
        test_accuracy = classifier.clf.score(X_test, y_test)

        # Ensure that the model is not overfitting (train accuracy should not be excessively higher than test accuracy)
        assert (abs(train_accuracy - test_accuracy) < 0.2)  # Example threshold for overfitting

    def test_classifier_data_integrity(self, classifier):
        """
        Test the integrity of the dataset used by the classifier.
        We check if the data is correctly shaped and not corrupted.
        """
        # Ensure that the features are of the expected shape
        assert classifier.X.shape == (150, 4)  # 150 samples, 4 features each
        assert classifier.y.shape == (150,)  # 150 target labels

        # Ensure no NaN values in the dataset
        assert not any(map(lambda x: x is None, classifier.X.flatten()))
        assert not any(map(lambda x: x is None, classifier.y))

    def test_classifier_not_empty(self, classifier):
        """
        Test if the classifier is not empty and the data is populated.
        """
        # Check if the features (X) and labels (y) are non-empty
        assert len(classifier.X) > 0
        assert len(classifier.y) > 0

    def test_integration_train_predict(self, classifier):
        """
        Test the integration of training and prediction.
        This ensures that the model can go through the entire pipeline without issues.
        """
        classifier.train_model()  # Train the model

        # We are not concerned with specific outputs, but ensure that no errors are raised during the entire pipeline.
        # If train_model works without exceptions, the full workflow is considered passed.
