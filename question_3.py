import numpy as np
import pandas as pd



def delta_J_beta_0(df,  beta_0, beta_1, beta_2):
    x1 = df.iloc[:,0]
    x2 = df.iloc[:, 1]
    y = df.iloc[:, 2]

    return -2*np.sum( y - (beta_0 + beta_1*x1 + beta_2*x2))
def delta_j_beta_1(df, beta_0, beta_1, beta_2):
    x1 = df.iloc[:, 0]
    x2 = df.iloc[:, 1]
    y = df.iloc[:, 2]
    return -2*x1@(y - (beta_0 + beta_1*x1 + beta_2*x2))
def delta_j_beta_2(df, beta_0, beta_1, beta_2):
    x1 = df.iloc[:, 0]
    x2 = df.iloc[:, 1]
    y = df.iloc[:, 2]

    return -2*x2@(y - (beta_0 + beta_1*x1 + beta_2*x2))
    

def new_df(df, beta_0, beta_1, beta_2):
    x1 = df.iloc[:, 0]
    x2 = df.iloc[:, 1]

    df["y'"] =  beta_0 + beta_1*x1 + beta_2*x2
    return df 
def r_squared(df, y_pred):
    y = df.iloc[:, 2]
    y_mean = y.mean()
    return (np.sum(np.square(y_pred - y_mean)) / np.sum(np.square(y - y_mean)))
def main():
    df = pd.read_csv("question_3.csv")
    print(df.head(5))
    beta_0 = 0
    beta_1 = 0
    beta_2 = 0
     
    alpha = 0.001
    n = len(df)
    
    while True:
        #For convergence
        old_beta_0 = beta_0
        old_beta1 = beta_1
        old_beta_2 = beta_2

        #Normalize gradient
        j_beta_0 = delta_J_beta_0(df, beta_0, beta_1, beta_2)/n
        j_beta_1 = delta_j_beta_1(df, beta_0, beta_1, beta_2)/n
        j_beta_2 = delta_j_beta_2(df, beta_0, beta_1, beta_2)/n
        
        #Updating Coeffiencients
        beta_0 = beta_0 - alpha*j_beta_0
        beta_1 = beta_1 - alpha*j_beta_1
        beta_2 = beta_2 - alpha*j_beta_2
        

        if np.sqrt((old_beta_0 - beta_0)**2 +(old_beta1 - beta_1)**2 +  (old_beta_2 - beta_2)**2 ) < 1e-6:
            break 
    y_pred = new_df(df, beta_0, beta_1, beta_2 )["y'"]
    print(f"beta_0: {beta_0: .3f}") 
    print(f"beta_1: {beta_1:.3f}")
    print(f"beta_2: {beta_2: .3f}")
    print(f"R-2: {r_squared(df, y_pred): .3f}")
    
   


if __name__ == "__main__":
    main()