# import packeges
import numpy as np
from sklearn.svm import SVC

# create 2 array with all sessions from each region
def getRegionalData (data):
    X_pcx = []
    X_plcoa = []
    for sess in data['x_pcx']:
        X_pcx.append(np.array(data[sess[0]]))
    for sess in data['x_plcoa']:
        X_plcoa.append(np.array(data[sess[0]]))
    return X_pcx,X_plcoa


def build_dataset_with_holdout (X_pcx, X_plcoa, N , sess_index = 1 , rng = None):
    # labels : (0 = pcx , 1 = plcoa)
    if rng is None:
        rng = np.random.default_rng()
    X , y = [],[]
    X_held,y_held = [],[]
    

    # Exclude the k session
    if sess_index is not None:
        
        X_pcx_test_session = X_pcx[sess_index]
        X_pcx_train = X_pcx[:sess_index] + X_pcx[sess_index+1:]
        
        X_plcoa_test_session = X_plcoa[sess_index]
        X_plcoa_train = X_plcoa[:sess_index] + X_plcoa[sess_index+1:]
        # Building separate session for testing true population
        X_pcx_test_session = X_pcx_test_session.T
        X_plcoa_test_session = X_plcoa_test_session.T
        
        N = min (X_pcx_test_session.shape[1],X_plcoa_test_session.shape[1], N)
        
    else:
        X_pcx_train = X_pcx
        X_plcoa_train = X_plcoa
    
    for sess in X_pcx_train:
        sess = sess.T  # Trasposing into expected trials X Neurons format
        if sess.shape[1] < N:
            continue
        idx = rng.choice(sess.shape[1], N, replace=False)
        X.append(sess[:, idx])
        y.append(np.zeros(sess.shape[0]))

    for sess in X_plcoa_train:
        sess = sess.T  # Trasposing into expected trials X Neurons format
        if sess.shape[1] < N:
            continue
        idx = rng.choice(sess.shape[1], N, replace=False)
        X.append(sess[:, idx])
        y.append(np.ones(sess.shape[0]))
        
        
    if sess_index is not None:
        idx_pcx = rng.choice(X_pcx_test_session.shape[1], N, replace=False) 
        idx_plcoa = rng.choice(X_plcoa_test_session.shape[1], N, replace=False)
        X_held = np.vstack([
        X_pcx_test_session[:, idx_pcx],
        X_plcoa_test_session[:, idx_plcoa]])
    
        y_held = np.concatenate([
        np.zeros(X_pcx_test_session.shape[0]),
        np.ones(X_plcoa_test_session.shape[0])])
        return np.vstack(X), np.concatenate(y) , X_held, y_held
    
    return np.vstack(X), np.concatenate(y) , None , None

# rbf SVM classification

def singleSVMClassification (X,y,X_held = None ,y_held = None, test_size = 0.1, random_state = 42):

    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=42)
    
    svc_orig = SVC(kernel='rbf')
    svc_orig.fit(X_train, y_train)
    accuracy_orig = svc_orig.score(X_test,y_test)
    if X_held is not None or y_held is not None:
        accuracy_held = svc_orig.score(X_held,y_held)
        print(f'svc accuracy on true population: {accuracy_held * 100:.2f}%')
    else:
        accuracy_held = None
    print(f'svc accuracy on pseudo population: {accuracy_orig * 100:.2f}%')
    
    return accuracy_orig , accuracy_held