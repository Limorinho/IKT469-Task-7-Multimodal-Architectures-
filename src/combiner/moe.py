class Moe:
    def __init__(self, models):
        self.routes = models

    def predict(self, input_data):
        predictions = []
        for model in self.routes:
            prediction = model.predict(input_data)
            predictions.append(prediction)
        return predictions
