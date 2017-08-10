
class RuleClassifier:
    @staticmethod
    def parameters():
        return {}

    def __init__(self, rule):
        self.rule = rule

    def fit(self, X, y):
        pass

    def predict(self, X):
        return self.rule(X)

    def __repr__(self):
        import inspect
        rule = inspect.getsource(self.rule).strip()
        return "<%s>" % rule
