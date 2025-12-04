
p_window_smps = [10, 15, 20, 25, 30]  

t_window_secs = [10, 15, 30, 60]  

test_window_secs_list = [x for x in range(1,11)]  

file_csv = "2024-07-21_07-31_ELIMINATION_PHASE_UBC_Thunderbots-vs-ITAndroids.log.txt.csv"
X_max    = 3300
Y_max    = 5000

input_features = [
    'rb0_x',  'rb0_y',
    'rb1_x',  'rb1_y',
    'rb2_x',  'rb2_y',
    'ry0_x',  'ry0_y',
    'ry1_x',  'ry1_y',
    'ry2_x',  'ry2_y',
    'ball_x', 'ball_y'
]
output_features = ['rb1_x', 'rb1_y']

import os
from time import time
import numpy as np
from pandas import read_csv
from numpy.lib.stride_tricks import sliding_window_view as slide_w
from scipy.stats import mode
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


def obter_dados(csv_file, X_max, Y_max):
    df = read_csv(csv_file)
    df.interpolate(method='linear', inplace=True)
    t_secs = df['t_capture'] - df['t_capture'][0]
    for k in df.keys():
        if k[-1]=='x':
            df[k] /= X_max
        elif k[-1]=='y':
            df[k] /= Y_max
    freq_hz = 1/mode(np.diff(df['t_capture']))[0]
    return df, t_secs, freq_hz

def testar_janela(df_cur, pred_window, train_window):
    X = np.zeros((len(df_cur) - pred_window, len(input_features)*pred_window))
    y = np.zeros((len(df_cur) - pred_window, len(output_features)))
    for k, f in enumerate(input_features):
        kk = k*pred_window
        X[:, kk:kk+pred_window] = slide_w(df_cur[f][0:-1], pred_window)
    for k,f in enumerate(output_features):
        y[:,k] = df_cur[f][pred_window:]
    X_train = X[:train_window,:]
    y_train = y[:train_window,:]
    X_test  = X[train_window:,:]
    y_test  = y[train_window:,:]
    
    t = [time()]
    model = LinearRegression().fit(X_train, y_train)
    t.append(time())
    y_pred = model.predict(X_test)
    t.append(time())
    
    train_time = t[1]-t[0]
    test_time  = t[2]-t[1]
    return y_test, y_pred, train_time, test_time

def testar_partida(df, pred_window, train_window_smps, test_window_smps):
    y_test = np.zeros((len(df),2))
    y_pred = np.zeros((len(df),2))
    train_time_list = []
    test_time_list  = []
    jmp = train_window_smps + test_window_smps + pred_window
    for k0 in range(0, len(df) - jmp, jmp):
        k1 = k0 + train_window_smps + pred_window
        k2 = k0 + jmp
        df_cur = df.iloc[k0:k2]
        yt, yp, trt, tst = testar_janela(df_cur, pred_window, train_window_smps)
        y_test[k1:k2] = yt
        y_pred[k1:k2] = yp
        train_time_list.append(trt)
        test_time_list.append(tst)
    return y_test, y_pred, train_time_list, test_time_list


ROOT_OUTPUT_DIR = "resultados"
os.makedirs(ROOT_OUTPUT_DIR, exist_ok=True)


df, t_secs, freq_hz = obter_dados(file_csv, X_max, Y_max)

resultados = []

for pred_window_smps in p_window_smps:
    for train_window_secs in t_window_secs:
        train_window_smps = int(freq_hz * train_window_secs)
        
        for test_window_secs in test_window_secs_list:
            test_window_smps  = int(freq_hz * test_window_secs)
            
            folder_name = f"pred{pred_window_smps}_train{train_window_secs}_test{test_window_secs}"
            output_dir = os.path.join(ROOT_OUTPUT_DIR, folder_name)
            os.makedirs(output_dir, exist_ok=True)
            
            y_test, y_pred, train_time, test_time = testar_partida(
                df, pred_window_smps, train_window_smps, test_window_smps)
            
            # Desnormalizar
            y_test[:,0] *= X_max
            y_test[:,1] *= Y_max
            y_pred[:,0] *= X_max
            y_pred[:,1] *= Y_max
            
            # Erros
            erro_eucl = np.sqrt((y_test[:,0]-y_pred[:,0])**2 + (y_test[:,1]-y_pred[:,1])**2)
            erro_eucl = erro_eucl[erro_eucl>0]
            res_x = y_test[:,0] - y_pred[:,0]
            res_y = y_test[:,1] - y_pred[:,1]
            rmse = np.sqrt(np.mean(erro_eucl**2))
            mae_x = np.mean(np.abs(res_x))
            mae_y = np.mean(np.abs(res_y))
            med_tr = np.median(train_time)
            
            resultados.append({
                'pred_window': pred_window_smps,
                'train_window': train_window_secs,
                'test_window': test_window_secs,
                'mean_erro_eucl_mm': np.mean(erro_eucl),
                'RMSE_mm': rmse,
                'MAE_X': mae_x,
                'MAE_Y': mae_y,
                'train_time_ms': med_tr*1000,
                'test_time_ms': np.median(test_time)*1000
            })

            output_metrics_filename = os.path.join(ROOT_OUTPUT_DIR, "../metrics.csv")
            if not os.path.exists(output_metrics_filename):
                with open(output_metrics_filename, "w") as f:
                    f.write("pred_window,train_window,test_window,mean_erro_eucl_mm,RMSE_mm,MAE_X,MAE_Y,train_time_ms,test_time_ms\n")
            with open(output_metrics_filename, "a") as f:
                f.write(f"{pred_window_smps},{train_window_secs},{test_window_secs},{np.mean(erro_eucl):.3f},{rmse:.3f},{mae_x:.3f},{mae_y:.3f},{med_tr*1000:.3f},{np.median(test_time)*1000:.3f}\n")
            
            output_filename = os.path.join(output_dir, "resultados.txt")
            with open(output_filename, "w") as f:
                f.write(f"RMSE(erro euclidiano) = {rmse:.2f} mm\n")
                f.write(f"MAE X = {mae_x:.2f} mm\n")
                f.write(f"MAE Y = {mae_y:.2f} mm\n")
                f.write(f"Tempo mediano de treinamento = {med_tr*1000:.2f} ms\n")
            
            plt.plot(y_test[:,0], y_test[:,1], '.', label='Real')
            plt.plot(y_pred[:,0], y_pred[:,1], '.', label='Predito')
            plt.legend(); plt.title('Trajetória da Rb1'); plt.xlabel('X'); plt.ylabel('Y'); plt.grid(True)
            plt.savefig(os.path.join(output_dir, "trajetoria_rb1.png"), dpi=300)
            plt.close()
            
            plt.semilogy(t_secs, np.abs(y_test[:,0]-y_pred[:,0]), '.', label='X')
            plt.semilogy(t_secs, np.abs(y_test[:,1]-y_pred[:,1]), '.', label='Y')
            plt.xlabel('Tempo (s)'); plt.ylabel('Erro absoluto (mm)'); plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(output_dir, "erro_absoluto.png"), dpi=300)
            plt.close()
            
            plt.hist(erro_eucl, bins=128, density=True, log=True)
            plt.xlabel('Erro euclidiano (mm)'); plt.ylabel('Probabilidade'); plt.title('Distribuição do erro'); plt.grid(True)
            plt.savefig(os.path.join(output_dir, "distribuicao_erro.png"), dpi=300)
            plt.close()
            
            plt.hist(train_time, bins=32, density=True)
            plt.xlabel('Tempo de treino (s)'); plt.ylabel('Probabilidade'); plt.title('Distribuição tempo de treino'); plt.grid(True)
            plt.savefig(os.path.join(output_dir, "distribuicao_tempo_treino.png"), dpi=300)
            plt.close()
            
            print(f"Finalizado pred={pred_window_smps}, train={train_window_secs}, test={test_window_secs}")

