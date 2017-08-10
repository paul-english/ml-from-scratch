from xgboost.sklearn import XGBRegressor


class WrappedXGBRegressor(XGBRegressor):
    # We wrap this so we can add the parameters property
    # for automatic CV/hyperparameter search
    @staticmethod
    def parameters():
        return {
            'n_estimators': [50, 200, 2000],
            # TODO enumerate these
        }
