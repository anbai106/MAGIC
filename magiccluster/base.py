import abc
import pandas as pd
from utils import GLMcorrection
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

__author__ = "Junhao Wen"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Junhao Wen, Erdem Varol"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Junhao Wen"
__email__ = "junhao.wen89@gmail.com"
__status__ = "Development"


class WorkFlow:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def run(self):
        pass


class Input:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_x(self):
        pass

    @abc.abstractmethod
    def get_y(self):
        pass

class OPNMF_Input(Input):

    def __init__(self, opnmf_dir, participant_tsv, covariate_tsv=None):
        self._opnmf_dir = opnmf_dir
        self._participant_tsv = participant_tsv
        self._covariate_tsv = covariate_tsv
        self._x = None
        self._y = None

        ## check the participant_tsv & covariate_tsv, the header, the order of the columns, etc
        self._df_feature = pd.read_csv(participant_tsv, sep='\t')
        if ('participant_id' != list(self._df_feature.columns.values)[0]) or (
                'session_id' != list(self._df_feature.columns.values)[1]) or \
                ('diagnosis' != list(self._df_feature.columns.values)[2]):
            raise Exception("the data file is not in the correct format."
                            "Columns should include ['participant_id', 'session_id', 'diagnosis']")
        self._subjects = list(self._df_feature['participant_id'])
        self._sessions = list(self._df_feature['session_id'])
        self._diagnosis = list(self._df_feature['diagnosis'])

    def get_x(self, num_component, opnmf_dir):

        ## alternatively, we use here the output of pyOPNMF loading coefficient
        loading_coefficient_csv = os.path.join(opnmf_dir, 'NMF', 'component_' + str(num_component),
                                               'loading_coefficient.tsv')
        ## read the tsv
        df_opnmf = pd.read_csv(loading_coefficient_csv, sep='\t')
        df_opnmf = df_opnmf.loc[df_opnmf['participant_id'].isin(self._df_feature['participant_id'])]
        ### adjust the order of the rows to match the original tsv files
        df_opnmf = df_opnmf.set_index('participant_id')
        df_opnmf = df_opnmf.reindex(index=self._df_feature['participant_id'])
        df_opnmf = df_opnmf.reset_index()

        self._x = df_opnmf[['component_' + str(i + 1) for i in range(num_component)]].to_numpy()

        ### normalize the data, note the normalization should be done for each component, not across component
        scaler = StandardScaler()
        self._x = scaler.fit_transform(self._x)

        if self._covariate_tsv is not None:
            df_covariate = pd.read_csv(self._covariate_tsv, sep='\t')
            if ('participant_id' != list(self._df_feature.columns.values)[0]) or (
                    'session_id' != list(self._df_feature.columns.values)[1]) or \
                    ('diagnosis' != list(self._df_feature.columns.values)[2]):
                raise Exception("the data file is not in the correct format."
                                "Columns should include ['participant_id', 'session_id', 'diagnosis']")
            participant_covariate = list(df_covariate['participant_id'])
            session_covariate = list(df_covariate['session_id'])
            label_covariate = list(df_covariate['diagnosis'])

            # check that the participant_tsv and covariate_tsv have the same orders for the first three column
            if (not self._subjects == participant_covariate) or (not self._sessions == session_covariate) or (
                    not self._diagnosis == label_covariate):
                raise Exception(
                    "the first three columns in the feature csv and covariate csv should be exactly the same.")

            ## normalize the covariate z-scoring
            data_covariate = df_covariate.iloc[:, 3:]
            data_covariate = ((data_covariate - data_covariate.mean()) / data_covariate.std()).values

            ## correction for the covariate, only retain the pathodological correspondance
            self._x, _ = GLMcorrection(self._x, np.asarray(self._diagnosis), data_covariate, self._x, data_covariate)

        return self._x

    def get_y(self):
        """
        Do not change the label's representation
        :return:
        """

        if self._y is not None:
            return self._y

        self._y = np.array(self._diagnosis)
        return self._y

