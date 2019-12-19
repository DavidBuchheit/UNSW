import os
from sklearn import metrics
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation


train_set = 'UNSW_NB15_training-set.csv'
test_set = 'UNSW_NB15_testing-set.csv'

training = pd.read_csv(train_set, index_col='id')
test = pd.read_csv(test_set, index_col='id')

def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd


# Encode text values to one hot encoding (i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)



CATEGORICAL_COLUMNS = ['proto', 'service', 'state', 'label']
NUMERIC_COLUMNS = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload',
                   'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb',
                   'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
                   'response_body_len', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',
                   'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
                   'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_sm_ips_ports',]

for col in CATEGORICAL_COLUMNS:
    encode_text_dummy(training, col)

for col in NUMERIC_COLUMNS:
    try:
        encode_numeric_zscore(training, col)
    except:
        print(col)

normal_mask = training['attack_cat'] == 'Normal'
attack_mask = training['attack_cat'] != 'Normal'


training.drop('attack_cat', axis=1, inplace=True)

df_normal = training[normal_mask]
df_attack = training[attack_mask]


print(f"Normal count: {len(df_normal)}")
print(f"Attack count: {len(df_attack)}")

x_normal = df_normal.values
x_attack = df_attack.values

from sklearn.model_selection import train_test_split

x_normal_train, x_normal_test = train_test_split(
    x_normal, test_size=0.25, random_state=42)

# %%

print(f"Normal train count: {len(x_normal_train)}")
print(f"Normal test count: {len(x_normal_test)}")

if os.path.exists("my_model.h5"):
    #Saving so we don't have to constantly train model when testing. Setting up data takes like no time so that can be ignored
    model = tf.keras.models.load_model('my_model.h5')
else:
    model = Sequential()
    model.add(Dense(64, input_dim=x_normal.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(x_normal.shape[1]))  # Multiple output neurons
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_normal_train, x_normal_train, verbose=1, epochs=100)

# %%

pred = model.predict(x_normal_test)
score1 = np.sqrt(metrics.mean_squared_error(pred, x_normal_test))
pred = model.predict(x_normal)
score2 = np.sqrt(metrics.mean_squared_error(pred, x_normal))
pred = model.predict(x_attack)
score3 = np.sqrt(metrics.mean_squared_error(pred, x_attack))
print(f"Insample Normal Score (RMSE): {score1}".format(score1))
print(f"Out of Sample Normal Score (RMSE): {score2}")
print(f"Attack Underway Score (RMSE): {score3}")

##Saving model

print(model.summary())
model.save('my_model.h5')

weights = model.get_weights()
print(weights)


### PRUNING
### Read up from https://arxiv.org/pdf/1812.02035.pdf
### https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras

##Removed while weights get figured out