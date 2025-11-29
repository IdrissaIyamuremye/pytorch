import numpy as np
import pandas as pd



def delta_J_beta_0(df,  beta_0, beta_1):
    x = df.iloc[:,0]
    y = df.iloc[:, 1]
    return -2*np.sum( y - (beta_0 + beta_1*x))
def delta_j_beta_1(df, beta_0, beta_1):
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    return -2*x@(y - (beta_0 + beta_1*x))

def y_linear(df, beta_0, beta_1):
    x1 = df.iloc[:, 0]

    df["y'"] =  beta_0 + beta_1*x1 
    return df 

def r_squared(df, y_pred):
    y = df.iloc[:, 1]
    y_mean = y.mean()
    return (np.sum(np.square(y_pred - y_mean)) / np.sum(np.square(y - y_mean)))
def main():
    df = pd.read_csv("question_2.csv")
    beta_0 = 0
    beta_1 = 0
     
    alpha = 0.001
    n = len(df)
    
    while True:
        #For convergence
        old_beta_0 = beta_0
        old_beta1 = beta_1

        #Normalize gradient
        j_beta_0 = delta_J_beta_0(df, beta_0, beta_1)/n
        j_beta_1 = delta_j_beta_1(df, beta_0, beta_1)/n
        
        #Updating Coeffiencients
        beta_0 = beta_0 - alpha*j_beta_0
        beta_1 = beta_1 - alpha*j_beta_1
        

        if np.sqrt((old_beta1-beta_1)**2 + (old_beta_0-beta_0)**2) < 1e-6:
            break 
     
    y_pred = y_linear(df, beta_0, beta_1)["y'"]

    print(f"beta_0: {beta_0: .3f}")
    print(f"beta_1: {beta_1: .3f}")
    print(f"R-2: {r_squared(df,y_pred): .3f}")


if __name__ == "__main__":
    main()