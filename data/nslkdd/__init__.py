import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

"""
NSL-KDD dataset: https://www.unb.ca/cic/datasets/nsl.html

@attribute 'duration' real
@attribute 'protocol_type' {'tcp','udp', 'icmp'} 
@attribute 'service' {'aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50'} 
@attribute 'flag' { 'OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH' }
@attribute 'src_bytes' real
@attribute 'dst_bytes' real
@attribute 'land' {'0', '1'}
@attribute 'wrong_fragment' real
@attribute 'urgent' real
@attribute 'hot' real
@attribute 'num_failed_logins' real
@attribute 'logged_in' {'0', '1'}
@attribute 'num_compromised' real
@attribute 'root_shell' real
@attribute 'su_attempted' real
@attribute 'num_root' real
@attribute 'num_file_creations' real
@attribute 'num_shells' real
@attribute 'num_access_files' real
@attribute 'num_outbound_cmds' real
@attribute 'is_host_login' {'0', '1'}
@attribute 'is_guest_login' {'0', '1'}
@attribute 'count' real
@attribute 'srv_count' real
@attribute 'serror_rate' real
@attribute 'srv_serror_rate' real
@attribute 'rerror_rate' real
@attribute 'srv_rerror_rate' real
@attribute 'same_srv_rate' real
@attribute 'diff_srv_rate' real
@attribute 'srv_diff_host_rate' real
@attribute 'dst_host_count' real
@attribute 'dst_host_srv_count' real
@attribute 'dst_host_same_srv_rate' real
@attribute 'dst_host_diff_srv_rate' real
@attribute 'dst_host_same_src_port_rate' real
@attribute 'dst_host_srv_diff_host_rate' real
@attribute 'dst_host_serror_rate' real
@attribute 'dst_host_srv_serror_rate' real
@attribute 'dst_host_rerror_rate' real
@attribute 'dst_host_srv_rerror_rate' real
@attribute 'class' {'normal', 'anomaly'}
"""

COLUMN_NAMES = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'class',
    'label'
]

CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']

DEFAULT_TRAIN_FILE_PATH = "data/nslkdd/KDDTrain+.txt"
DEFAULT_TRAIN_FILE_PATH_20_PERCENT = "data/nslkdd/KDDTrain+_20Percent.txt"
DEFAULT_TEST_FILE_PATH = "data/nslkdd/KDDTest+.txt"
DEFAULT_TEST_FILE_PATH_20_PERCENT = "data/nslkdd/KDDTest-21.txt"


class NSLKDD_Preprocessor:
    def __init__(self):
        """
        Preprocessor for NSL-KDD dataset used by Choi et al. (2019).
        """
        
        self.categorical_cols = CATEGORICAL_FEATURES
        self.numerical_cols = [col for col in COLUMN_NAMES if col not in CATEGORICAL_FEATURES+['class', 'label']]
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler = MinMaxScaler()

        self.is_fitted = False

    def fit(self, X: pd.DataFrame):
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a Pandas DataFrame")

        transformed_data = []

        if self.categorical_cols:
            self.encoder.fit(X[self.categorical_cols])
            self.encoder_feature_names = self.encoder.get_feature_names_out(self.categorical_cols)
        
        if self.numerical_cols:
            self.scaler.fit(X[self.numerical_cols])

        self.is_fitted = True

    
    def transform(self, X: pd.DataFrame):
        if not self.is_fitted:
            raise ValueError('Preprocessor is not fitted')

        transformed_data = []

        # One-hot encoding
        if self.categorical_cols:
            cat_transformed = self.encoder.transform(X[self.categorical_cols])
            cat_transformed_df = pd.DataFrame(cat_transformed, columns=self.encoder_feature_names, index=X.index)
            transformed_data.append(cat_transformed_df)

        # MinMax scaling
        if self.numerical_cols:
            num_transformed = self.scaler.transform(X[self.numerical_cols])
            num_transformed_df = pd.DataFrame(num_transformed, columns=self.numerical_cols, index=X.index)
            transformed_data.append(num_transformed_df)

        X = pd.concat(transformed_data, axis=1)

        return X
    
    def fit_transform(self, X: pd.DataFrame):
        self.fit(X)
        return self.transform(X)
    


def load_nslkdd(filepath: str=None, random_20_percent: bool=True, partition: str='train') -> tuple[pd.DataFrame, pd.Series]:
    if filepath is None:
        if partition == 'train':
            filepath = DEFAULT_TRAIN_FILE_PATH_20_PERCENT if random_20_percent else DEFAULT_TRAIN_FILE_PATH
        else:
            filepath = DEFAULT_TEST_FILE_PATH_20_PERCENT if random_20_percent else DEFAULT_TEST_FILE_PATH

    df = pd.read_csv(filepath, names=COLUMN_NAMES)
    X = df.drop(columns=['class', 'label'])
    y = df['class']

    return X, y