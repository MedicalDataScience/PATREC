
from sklearn.ensemble import RandomForestClassifier


from learning.BaseClassifier import BaseClassifier


class ClassifierRF(BaseClassifier):

    def __init__(self, options):
        BaseClassifier.__init__(self, 'rf');

        self.n_estimators = options['n_estimators'];
        self.max_depth = options['max_depth'];
        self.random_state = options['random_state'];
        self.class_weight = options['class_weight'];
        self.n_jobs = options['n_jobs'];

        self.clf = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state, n_jobs=self.n_jobs, class_weight=self.class_weight);
        return;