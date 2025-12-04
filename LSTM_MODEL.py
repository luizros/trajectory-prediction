# %%
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import sys
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import time
from scipy import stats

tf.random.set_seed(42)
np.random.seed(42)
plt.style.use('default')

sys.path.append('../')
from utils.createSequence import create_sequences
from utils.drawField import drawField, plot_trajectory
from utils.lstm_create_model import create_lstm_model
from utils.getSampleRate import getSampleRate
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


N_STEPS = 5*60
MAX_SAMPLES = 47000 
PREDICTION_HORIZON = 60


# %%
class MatchAnalyzer:
    def __init__(self, csv_filepath):
        if not os.path.exists(csv_filepath):
            raise FileNotFoundError(f"Arquivo não encontrado: {csv_filepath}")
        
        self.filepath = csv_filepath
        self.match_name = os.path.basename(csv_filepath).replace('.log.txt.csv', '')
        self.df = pd.read_csv(csv_filepath)
        self.num_frames = self.df.shape[0]
        self.sampling_rate = None
        self.missing_data_report = None
        
        print(f"Math: {self.match_name} ({self.num_frames} frames)")

    # Calcula a taxa de amostragem com base no t_capture
    def _analyze_sampling_rate(self, time_column='t_capture'):
        try:
            self.sampling_rate = getSampleRate(self.df, time_column=time_column)
        except:
            self.sampling_rate = 60.0  

    def _check_missing_values(self):
        nulls_per_column = self.df.isnull().sum()
        self.missing_data_report = nulls_per_column[nulls_per_column > 0].sort_values(ascending=False)

    def run_analysis(self):
        self._analyze_sampling_rate()
        self._check_missing_values()
        print(f"Análise concluída para {self.match_name}")

    def getTimeMatch(self):
        if self.sampling_rate is None:
            self.run_analysis()
        total_time_seconds = self.num_frames / self.sampling_rate if self.sampling_rate else 0
        return total_time_seconds   

    def get_summary_dict(self):
        if self.sampling_rate is None:
            self.run_analysis()
        return {
            'Nome da Partida': self.match_name,
            'Caminho do Arquivo': self.filepath,
            'Robos azuis': len([col for col in self.df.columns if col.startswith('rb') and col.endswith('_x')]),
            'Robos amarelos': len([col for col in self.df.columns if col.startswith('ry') and col.endswith('_x')]),
            'Duração Total (s)': self.getTimeMatch(),
            'Total de Frames': self.num_frames,
            'Taxa de Amostragem (Hz)': f"{self.sampling_rate:.2f}" if self.sampling_rate else "N/A",
            'Colunas com Dados Faltando': len(self.missing_data_report) if self.missing_data_report is not None else 0,
            'Total de Células Faltando': int(self.missing_data_report.sum()) if self.missing_data_report is not None else 0
        }

# %%
csv_files = glob.glob('../../datasets copy/csv/*.csv')

partidas_analisadas = []
for file in csv_files:
    match = MatchAnalyzer(file)
    match.run_analysis()
    partidas_analisadas.append(match)


# %%
lista_de_resumos = [p.get_summary_dict() for p in partidas_analisadas]
df_resumo_completo = pd.DataFrame(lista_de_resumos)

# %%
partida_info = df_resumo_completo.sort_values(by='Total de Células Faltando').iloc[0]
match_filepath = partida_info['Caminho do Arquivo']

print(f"Partida selecionada: {partida_info['Nome da Partida']}")
print(f"Total de frames: {partida_info['Total de Frames']:,}")

df_raw = pd.read_csv(match_filepath)

colunas_essenciais = [
    't_capture',
    'rb0_x', 'rb0_y', 'rb0_theta', 'rb1_x', 'rb1_y', 'rb1_theta', 'rb2_x', 'rb2_y', 'rb2_theta',
    'ry0_x', 'ry0_y', 'ry0_theta', 'ry1_x', 'ry1_y', 'ry1_theta', 'ry2_x', 'ry2_y', 'ry2_theta', 
    'ball_x', 'ball_y'
]

colunas_existentes = [col for col in colunas_essenciais if col in df_raw.columns]
df_filter = df_raw[colunas_existentes].copy()

print(f"Dados filtrados: {df_filter.shape}")

# %%
df_filter.head()

# %%
# Check missing data 
missing_data = df_filter.isnull().sum()
missing_data = missing_data[missing_data > 0]
print("Collums with no data:")
print(missing_data)

# %%
missing_data_percentage = (missing_data / len(df_filter)) * 100
print("Percentage of missing data per column:")
print(f'{missing_data_percentage} %')

# %%
critical_columns = missing_data_percentage[missing_data_percentage > 5].index.tolist()
print("Critical columns (more than 5% missing data):")
print(critical_columns)

# %%
def handle_missing_data(df, method='interpolate'):
    df_copy = df.copy()
    
    if method == 'interpolate':
        df_copy.interpolate(method='linear', inplace=True)
    elif method == 'drop':
        df_copy.dropna(inplace=True)
    elif method == 'moving_average':
        for col in df_copy.columns:
            if df_copy[col].isnull().any():
                df_copy[col] = df_copy[col].fillna(df_copy[col].rolling(window=5, min_periods=1).mean())
    else:
        raise ValueError("Método desconhecido. Use 'interpolate', 'drop' ou 'moving_average'.")
    
    return df_copy

# %%
df_interpolated = handle_missing_data(df_filter, method='interpolate')

# %%
missing_after = df_interpolated.isnull().sum()
print("Missing data after handling:")
print(missing_after[missing_after > 0])

# %%
TOTAL_FRAMES = len(df_interpolated)

# %%
FRAMES_TO_PLOT = TOTAL_FRAMES  
START_FRAME = 0 
df_viz = df_interpolated.iloc[START_FRAME:START_FRAME + FRAMES_TO_PLOT].reset_index(drop=True)

df_viz['elapsed_time'] = df_viz['t_capture'] - df_viz['t_capture'].iloc[0]

time_range = df_viz['elapsed_time'].iloc[-1]
print(f"Range: [{len(df_viz)}] frames (de {START_FRAME} a {START_FRAME + len(df_viz)})\nTime: {time_range:.2f} segundos")

# %%
# Ball Plots Analisis
plt.plot(df_viz['elapsed_time'], df_viz['ball_x'], 
                label='Ball X', color='orange', linewidth=2)
plt.plot(df_viz['elapsed_time'], df_viz['ball_y'], 
                label='Ball Y', color='red', linewidth=2)
plt.title(f'Ball Trajectory - Frames {START_FRAME} to {FRAMES_TO_PLOT + START_FRAME}', fontsize=12, color= 'black',)
plt.xlabel('Tempo Decorrido (s)') # RÓTULO ATUALIZADO
plt.ylabel('Position (mm)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Ball statistics
print("Ball Statistics:")
ball_stats = f"Ball X: μ={df_viz['ball_x'].mean():.0f} σ={df_viz['ball_x'].std():.0f}"
ball_stats += f" Ball Y: μ={df_viz['ball_y'].mean():.0f} σ={df_viz['ball_y'].std():.0f}"
print(ball_stats)

# %%
# Robot blue 0 Plots Analisis
plt.plot(df_viz['elapsed_time'], df_viz['rb0_x'], 
                label='Robot blue 0 X', color='orange', linewidth=2)
plt.plot(df_viz['elapsed_time'], df_viz['rb0_y'], 
                label='Robot blue 0 Y', color='red', linewidth=2)
plt.title(f'Robot blue 0 Trajectory - Frames {START_FRAME} to {FRAMES_TO_PLOT + START_FRAME}', fontsize=12, color= 'black',)
plt.ylabel('Position (mm)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# Robot blue 0 statistics
print("Robot blue 0 Statistics:")
robot_stats = f"Robot blue 0 X: μ={df_viz['rb0_x'].mean():.0f} σ={df_viz['rb0_x'].std():.0f}"
robot_stats += f" Robot blue 0 Y: μ={df_viz['rb0_y'].mean():.0f} σ={df_viz['rb0_y'].std():.0f}"
print(robot_stats)


# %%
# Robot blue 1 Plots Analisis
plt.plot(df_viz['elapsed_time'], df_viz['rb1_x'], 
                label='Robot blue 1 X', color='orange', linewidth=2)
plt.plot(df_viz['elapsed_time'], df_viz['rb1_y'], 
                label='Robot blue 1 Y', color='red', linewidth=2)
plt.title(f'Robot blue 1 Trajectory - Frames {START_FRAME} to {FRAMES_TO_PLOT + START_FRAME}', fontsize=12, color= 'black',)
plt.ylabel('Position (mm)')
plt.legend()
plt.grid(True, alpha=0.3, color='blue')
plt.show()
# Robot blue 1 statistics
print("Robot blue 1 Statistics:")
robot_stats = f"Robot blue 1 X: μ={df_viz['rb1_x'].mean():.0f} σ={df_viz['rb1_x'].std():.0f}"
robot_stats += f" Robot blue 1 Y: μ={df_viz['rb1_y'].mean():.0f} σ={df_viz['rb1_y'].std():.0f}"
print(robot_stats)

# %%
# Robot blue 2 Plots Analisis
plt.plot(df_viz['elapsed_time'], df_viz['rb2_x'], 
                label='Robot blue 2 X', color='orange', linewidth=2)
plt.plot(df_viz['elapsed_time'], df_viz['rb2_y'], 
                label='Robot blue 2 Y', color='red', linewidth=2)
plt.title(f'Robot blue 2 Trajectory - Frames {START_FRAME} to {FRAMES_TO_PLOT + START_FRAME}', fontsize=12, color= 'black',)
plt.ylabel('Position (mm)')
plt.legend()
plt.grid(True, alpha=0.3, color='blue')
plt.show()
# Robot blue 2 statistics
print("Robot blue 2 Statistics:")
robot_stats = f"Robot blue 2 X: μ={df_viz['rb2_x'].mean():.0f} σ={df_viz['rb2_x'].std():.0f}"
robot_stats += f" Robot blue 2 Y: μ={df_viz['rb2_y'].mean():.0f} σ={df_viz['rb2_y'].std():.0f}"
print(robot_stats)

# %%
# Robot yellow 0 Plots Analisis
plt.plot(df_viz['elapsed_time'], df_viz['ry0_x'], 
                label='Robot yellow 0 X', color='orange', linewidth=2)
plt.plot(df_viz['elapsed_time'], df_viz['ry0_y'], 
                label='Robot yellow 0 Y', color='red', linewidth=2)
plt.title(f'Robot yellow 0 Trajectory - Frames {START_FRAME} to {FRAMES_TO_PLOT + START_FRAME}', fontsize=12, color= 'black',)
plt.xlabel('Time (s)')
plt.ylabel('Position (mm)')
plt.legend()
plt.grid(True, alpha=0.3, color='yellow')
plt.show()
# Robot yellow 0 statistics
print("Robot yellow 0 Statistics:")
robot_stats = f"Robot yellow 0 X: μ={df_viz['ry0_x'].mean():.0f} σ={df_viz['ry0_x'].std():.0f}"
robot_stats += f" Robot yellow 0 Y: μ={df_viz['ry0_y'].mean():.0f} σ={df_viz['ry0_y'].std():.0f}"
print(robot_stats)

# %%
# Robot yellow 1 Plots Analisis
plt.plot(df_viz['elapsed_time'], df_viz['ry1_x'], 
                label='Robot yellow 1 X', color='orange', linewidth=2)
plt.plot(df_viz['elapsed_time'], df_viz['ry1_y'], 
                label='Robot yellow 1 Y', color='red', linewidth=2)
plt.title('Robot yellow 1 Trajectory', fontsize=14, fontweight='bold', color= 'black',)
plt.xlabel('Time (s)')
plt.ylabel('Position (mm)')
plt.legend()
plt.grid(True, alpha=0.3, color='yellow')
plt.show()
# Robot yellow 1 statistics
print("Robot yellow 1 Statistics:")
robot_stats = f"Robot yellow 1 X: μ={df_viz['ry1_x'].mean():.0f} σ={df_viz['ry1_x'].std():.0f}"
robot_stats += f" Robot yellow 1 Y: μ={df_viz['ry1_y'].mean():.0f} σ={df_viz['ry1_y'].std():.0f}"
print(robot_stats)

# %%
# Robot yellow 2 Plots Analisis
plt.plot(df_viz['elapsed_time'], df_viz['ry2_x'], 
                label='Robot yellow 2 X', color='orange', linewidth=2)
plt.plot(df_viz['elapsed_time'], df_viz['ry2_y'], 
                label='Robot yellow 2 Y', color='red', linewidth=2)
plt.title('Robot yellow 2 Trajectory', fontsize=14, fontweight='bold', color= 'black',)
plt.xlabel('Time (s)')
plt.ylabel('Position (mm)')
plt.legend()
plt.grid(True, alpha=0.3, color='yellow')
plt.show()
# Robot yellow 2 statistics
print("Robot yellow 2 Statistics:")
robot_stats = f"Robot yellow 2 X: μ={df_viz['ry2_x'].mean():.0f} σ={df_viz['ry2_x'].std():.0f}"
robot_stats += f" Robot yellow 2 Y: μ={df_viz['ry2_y'].mean():.0f} σ={df_viz['ry2_y'].std():.0f}"
print(robot_stats)

# %%
df_filter = df_interpolated[df_interpolated['t_capture'] >= df_interpolated['t_capture'].iloc[0] + 8.0].reset_index(drop=True)
df_filter['elapsed_time'] = df_filter['t_capture'] - df_filter['t_capture'].iloc[0]

FRAMES_TO_PLOT = 1000  
START_FRAME = 0       
df_viz = df_filter.iloc[START_FRAME:START_FRAME + FRAMES_TO_PLOT].reset_index(drop=True)

df_viz['elapsed_time'] = df_viz['t_capture'] - df_viz['t_capture'].iloc[0]

time_range = df_viz['elapsed_time'].iloc[-1]
print(f"Range: [{len(df_viz)}] frames (de {START_FRAME} a {START_FRAME + len(df_viz)})\nTime: {time_range:.2f} segundos")

plt.plot(df_viz['elapsed_time'] - df_viz['elapsed_time'].iloc[0], df_viz['ball_x'], 
                label='Ball X', color='orange', linewidth=2)
plt.plot(df_viz['elapsed_time'] - df_viz['elapsed_time'].iloc[0], df_viz['ball_y'], 
                label='Ball Y', color='red', linewidth=2)
plt.title('Filtered Ball Trajectory (after 8s)', fontsize=14, fontweight='bold', color= 'black',)
plt.xlabel('Elapsed Time (s)')
plt.ylabel('Position (mm)')
plt.legend()
plt.grid(True, alpha=0.3)

# %%
fig, ax = drawField()
plot_trajectory(ax, df_viz['ball_x'], df_viz['ball_y'], label='Ball', color='red')
plot_trajectory(ax, df_viz['rb0_x'], df_viz['rb0_y'], label='Robot Blue 0', color='blue')
plot_trajectory(ax, df_viz['rb1_x'], df_viz['rb1_y'], label='Robot Blue 1', color='blue')

# %%
df_work = df_filter.copy()
print(f"Dados de trabalho: {df_work.shape}")

df_work['ball_dx'] = df_work['ball_x'].diff()
df_work['ball_dy'] = df_work['ball_y'].diff()
df_work['ball_speed'] = np.sqrt(df_work['ball_dx']**2 + df_work['ball_dy']**2)

MAX_SPEED_POSSIBLE = 1000  # mm/frame
df_work['is_continuous'] = (df_work['ball_speed'] < MAX_SPEED_POSSIBLE).astype(int)

df_work['ball_acceleration'] = df_work['ball_speed'].diff()


print(f"Velocidade média: {df_work['ball_speed'].mean():.1f} mm/frame")
print(f"Velocidade máxima: {df_work['ball_speed'].max():.1f} mm/frame")
print(f"% frames contínuos: {df_work['is_continuous'].mean()*100:.1f}%")

# %%


# %%


# %%


# %%
REALISTIC_MAX_SPEED = 300  # mm/frame (~18 m/s = 65 km/h)
print(f"Novo threshold: {REALISTIC_MAX_SPEED} mm/frame")

df_work['is_continuous'] = (df_work['ball_speed'] < REALISTIC_MAX_SPEED).astype(int)

outliers = df_work['ball_speed'] >= REALISTIC_MAX_SPEED
print(f"Outliers detectados: {outliers.sum()} frames")
print(f"% outliers: {(outliers.sum()/len(df_work))*100:.2f}%")
print(f"% frames contínuos: {df_work['is_continuous'].mean()*100:.1f}%")

window_size = 7  
df_work['ball_x_smooth'] = savgol_filter(df_work['ball_x'], window_size, 2)
df_work['ball_y_smooth'] = savgol_filter(df_work['ball_y'], window_size, 2)
df_work['rb0_x_smooth'] = savgol_filter(df_work['rb0_x'], window_size, 2)
df_work['rb0_y_smooth'] = savgol_filter(df_work['rb0_y'], window_size, 2)
df_work['rb1_x_smooth'] = savgol_filter(df_work['rb1_x'], window_size, 2)
df_work['rb1_y_smooth'] = savgol_filter(df_work['rb1_y'], window_size, 2)
df_work['rb2_x_smooth'] = savgol_filter(df_work['rb2_x'], window_size, 2)
df_work['rb2_y_smooth'] = savgol_filter(df_work['rb2_y'], window_size, 2)

df_work['ry0_x_smooth'] = savgol_filter(df_work['ry0_x'], window_size, 2)
df_work['ry0_y_smooth'] = savgol_filter(df_work['ry0_y'], window_size, 2)
df_work['ry1_x_smooth'] = savgol_filter(df_work['ry1_x'], window_size, 2)
df_work['ry1_y_smooth'] = savgol_filter(df_work['ry1_y'], window_size, 2)
df_work['ry2_x_smooth'] = savgol_filter(df_work['ry2_x'], window_size, 2)
df_work['ry2_y_smooth'] = savgol_filter(df_work['ry2_y'], window_size, 2)


print(f"Suavização aplicada (janela: {window_size})")
print(f"ETAPA 2 concluída - dados prontos para sequências!")


# %%
n_pontos = 50
# plotar x antes e depois da suavização
plt.plot(df_work['elapsed_time'][:n_pontos], df_work['ball_x'][:n_pontos], label='Ball X Original', color='orange', linewidth=1)
plt.plot(df_work['elapsed_time'][:n_pontos], df_work['ball_x_smooth'][:n_pontos], label='Ball X Suavizado', color='blue', linewidth=1)
plt.title('Ball X: Original vs Suavizado', fontsize=12, color= 'black',)
plt.xlabel('Tempo Decorrido (s)')
plt.ylabel('Position (mm)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
n_pontos = 20000
# plotar x antes e depois da suavização
plt.plot(df_work['elapsed_time'][:n_pontos], df_work['rb1_x'][:n_pontos], label='Rb1 X Original', color='orange', linewidth=1)
plt.plot(df_work['elapsed_time'][:n_pontos], df_work['rb1_x_smooth'][:n_pontos], label='Rb1 X Suavizado', color='blue', linewidth=1)
plt.title('Rb1 X: Original vs Suavizado', fontsize=12, color= 'black',)
plt.xlabel('Tempo Decorrido (s)')
plt.ylabel('Position (mm)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %%
# Adicionar física à bola
df_work['ball_vx'] = df_work['ball_x_smooth'].diff()  # Velocidade X
df_work['ball_vy'] = df_work['ball_y_smooth'].diff()  # Velocidade Y
df_work['ball_speed'] = np.sqrt(df_work['ball_vx']**2 + df_work['ball_vy']**2)
df_work['ball_acceleration'] = df_work['ball_speed'].diff()

# adicionar fisica aos robos rb1
df_work['rb1_vx'] = df_work['rb1_x_smooth'].diff()  # Velocidade X
df_work['rb1_vy'] = df_work['rb1_y_smooth'].diff()  # Velocidade Y
df_work['rb1_speed'] = np.sqrt(df_work['rb1_vx']**2 + df_work['rb1_vy']**2)
df_work['rb1_acceleration'] = df_work['rb1_speed'].diff()

# adicionar fisica aos robos rb0
df_work['rb0_vx'] = df_work['rb0_x_smooth'].diff()  # Velocidade X
df_work['rb0_vy'] = df_work['rb0_y_smooth'].diff()  # Velocidade Y
df_work['rb0_speed'] = np.sqrt(df_work['rb0_vx']**2 + df_work['rb0_vy']**2)
df_work['rb0_acceleration'] = df_work['rb0_speed'].diff()

# adicionar fisica aos robos rb2
df_work['rb2_vx'] = df_work['rb2_x_smooth'].diff()  # Velocidade X
df_work['rb2_vy'] = df_work['rb2_y_smooth'].diff()  # Velocidade Y
df_work['rb2_speed'] = np.sqrt(df_work['rb2_vx']**2 + df_work['rb2_vy']**2)
df_work['rb2_acceleration'] = df_work['rb2_speed'].diff()

# adicionar fisica aos robos ry1
df_work['ry1_vx'] = df_work['ry1_x_smooth'].diff()  # Velocidade X
df_work['ry1_vy'] = df_work['ry1_y_smooth'].diff()  # Velocidade Y
df_work['ry1_speed'] = np.sqrt(df_work['ry1_vx']**2 + df_work['ry1_vy']**2)
df_work['ry1_acceleration'] = df_work['ry1_speed'].diff()

# adicionar fisica aos robos ry0
df_work['ry0_vx'] = df_work['ry0_x_smooth'].diff()  # Velocidade X
df_work['ry0_vy'] = df_work['ry0_y_smooth'].diff()  # Velocidade Y
df_work['ry0_speed'] = np.sqrt(df_work['ry0_vx']**2 + df_work['ry0_vy']**2)
df_work['ry0_acceleration'] = df_work['ry0_speed'].diff()

# adicionar fisica aos robos ry2
df_work['ry2_vx'] = df_work['ry2_x_smooth'].diff()  # Velocidade X
df_work['ry2_vy'] = df_work['ry2_y_smooth'].diff()  # Velocidade Y
df_work['ry2_speed'] = np.sqrt(df_work['ry2_vx']**2 + df_work['ry2_vy']**2)
df_work['ry2_acceleration'] = df_work['ry2_speed'].diff()

#distancia do robô 1 até a bola
df_work['rb1_ball_distance'] = np.sqrt((df_work['rb1_x_smooth'] - df_work['ball_x_smooth'])**2 + (df_work['rb1_y_smooth'] - df_work['ball_y_smooth'])**2)



# # Input features melhoradas
# input_features = [
#     'ball_x_smooth', 'ball_y_smooth',           # Posição -> remover
#     'ball_vx', 'ball_vy',         # Velocidade
#     'ball_acceleration',           # Aceleração
#     'rb1_ball_distance',          # Distância do robô 1 até a bola

#     'ball_x', 'ball_y',           # Posição
#     # robos azuis
#     'rb1_x_smooth', 'rb1_y_smooth',
#     'rb1_vx', 'rb1_vy', 'rb1_speed', 'rb1_acceleration',                                         # Robot Blue 1
#     'rb0_x_smooth', 'rb0_y_smooth', 'rb0_vx', 'rb0_vy', 'rb0_speed', 'rb0_acceleration',                       # Robot Blue 0
#     'rb2_x_smooth', 'rb2_y_smooth', 'rb2_vx', 'rb2_vy', 'rb2_speed', 'rb2_acceleration',                       # Robot Blue 2

#     # # robos amarelos
#     # 'ry0_x', 'ry0_y', 'ry0_vx', 'ry0_vy', 'ry0_speed', 'ry0_acceleration',                     # Robot Yellow 0
#     # 'ry1_x', 'ry1_y', 'ry1_vx', 'ry1_vy', 'ry1_speed', 'ry1_acceleration',                     # Robot Yellow 1
#     # 'ry2_x', 'ry2_y', 'ry2_vx', 'ry2_vy', 'ry2_speed', 'ry2_acceleration'                      # Robot Yellow 2

# ]


input_features = [
    'rb1_x_smooth', 'rb1_y_smooth',
    # 'rb1_vx', 'rb1_vy', 'rb1_speed', 'rb1_acceleration',
    # 'ball_x_smooth', 'ball_y_smooth'
]



# input_features = [
#     # 'ball_x_smooth', 'ball_y_smooth',     # Bola (2 features)
#     'rb0_x', 'rb0_y',                     # Robot Blue 0
#     # 'rb1_x', 'rb1_y',                     # Robot Blue 1  
#     # 'rb2_x', 'rb2_y',                     # Robot Blue 2
#     # 'ry0_x', 'ry0_y',                     # Robot Yellow 0
#     # 'ry1_x', 'ry1_y',                     # Robot Yellow 1
#     # 'ry2_x', 'ry2_y'                      # Robot Yellow 2
# ]

# output_features = ['ball_x_smooth', 'ball_y_smooth'] # remover  usar suavisa
output_features = ['rb1_x_smooth', 'rb1_y_smooth'] 

print(f"INPUT Features ({len(input_features)}): {input_features}")
print(f"OUTPUT Features ({len(output_features)}): {output_features}")

df_final = df_work[input_features].copy()
df_output = df_work[output_features].copy()

print(f"Dados de entrada:")
print(f"Shape: {df_final.shape}")
print(f"Features: {len(input_features)}")

print(f"Dados de saída:")
print(f"Shape: {df_output.shape}")
print(f"Features: {len(output_features)}")

# %%
df_final.head()


df_final.fillna(0, inplace=True)
df_output.fillna(0, inplace=True)

# %%
# scaler_input = MinMaxScaler()
# data_input_normalized = scaler_input.fit_transform(df_final.copy())

# scaler_output = MinMaxScaler()
# data_output_normalized = scaler_output.fit_transform(df_output.copy())

from sklearn.preprocessing import StandardScaler

scaler_x = StandardScaler()
scaler_y = StandardScaler()

data_input_normalized = scaler_x.fit_transform(df_final)
data_output_normalized = scaler_y.fit_transform(df_output)


print(f"Input normalizado: {data_input_normalized.shape}")
print(f"Output normalizado: {data_output_normalized.shape}")
print(f"Features input: {len(input_features)}")
print(f"Features output: {len(output_features)}")

print(f"Normalização concluída:")
print(f"Entrada: {data_input_normalized.shape} (14 features)")
print(f"Saída: {data_output_normalized.shape} (2 features)")

# def create_sequences_custom(input_data, output_data, n_steps, max_samples=None):
#     X, y = [], []
    
#     if max_samples is not None and len(input_data) > max_samples:
#         input_data = input_data[:max_samples]
#         output_data = output_data[:max_samples]

#     for i in range(n_steps, len(input_data) - PREDICTION_HORIZON):
#         X.append(input_data[i-n_steps:i])
#         y.append(output_data[i + PREDICTION_HORIZON])
    
#     return np.array(X), np.array(y)


import numpy as np

def create_sequences_custom_fixed(input_data, output_data, n_steps, max_samples=None, prediction_horizon=0):
    """
    Cria janelas deslizantes (sequências) para treinamento de RNN/LSTM.
    
    :param input_data: Array NumPy dos dados de entrada (X_bruto).
    :param output_data: Array NumPy dos dados de saída/alvo (y_bruto).
    :param n_steps: Tamanho da janela de histórico (N_STEPS, ex: 5).
    :param max_samples: Número máximo de amostras (janelas) a serem criadas.
    :param prediction_horizon: Número de passos para frente a prever (0 = próximo passo imediato).
    :return: X (janelas de entrada) e y (alvos de saída).
    """
    X, y = [], []
    
    # 1. Calcular o número total de amostras que o dataset pode suportar.
    # O último índice possível para o alvo (y) é len(input_data) - 1.
    # O primeiro índice possível para a janela de entrada é n_steps.
    # O número total de amostras que cabem é:
    # len(input_data) - n_steps - prediction_horizon
    
    num_samples_possible = len(input_data) - n_steps - prediction_horizon
    
    # 2. Definir o número real de amostras a criar.
    # Se max_samples for fornecido, usaremos o menor entre ele e o possível.
    if max_samples is not None:
        num_samples_to_create = min(max_samples, num_samples_possible)
    else:
        num_samples_to_create = num_samples_possible
        
    # 3. Iterar e criar as janelas
    # A iteração vai de 0 até (num_samples_to_create - 1)
    for i in range(num_samples_to_create):
        
        # O início da janela (start_index)
        start_idx = i
        
        # O fim da janela (end_index), que é o ponto de corte do input_data [start_idx : end_idx]
        end_idx = start_idx + n_steps
        
        # O índice do alvo (target_idx), que está à frente da janela.
        target_idx = end_idx + prediction_horizon
        
        # Cria o par X, y
        X.append(input_data[start_idx:end_idx])
        y.append(output_data[target_idx])
    
    return np.array(X), np.array(y)

# %%
# se o robô ficar para

# %%

# Criar sequências
X, y = create_sequences_custom_fixed(
    input_data=data_input_normalized,
    output_data=data_output_normalized, 
    n_steps=N_STEPS,
    max_samples=MAX_SAMPLES,
    prediction_horizon=0

)


print(f"Sequências criadas:")
print(f"X shape: {X.shape} - (samples, timesteps, input_features)")
print(f"y shape: {y.shape} - (samples, output_features)")
print(f"Input features: {X.shape[2]}")
print(f"Output features: {y.shape[1]}")
print(f"Histórico: {N_STEPS} frames")
print(f"Amostras: {len(X)}")

# %%
N = int(len(X) * 0.7)  
X_train = X[:N]
y_train = y[:N]

X_test = X[N:]  
y_test = y[N:]

print(f"Train: X={X_train.shape}, y={y_train.shape}")
print(f"Test:  X={X_test.shape}, y={y_test.shape}")

# %%
print(f"Dimensões:")
print(f"Train: X={X_train.shape}, y={y_train.shape}")
print(f"Test:  X={X_test.shape}, y={y_test.shape}")
print(f"Input features: {X.shape[2]} (bola + robôs)")
print(f"Output features: {y.shape[1]} (apenas bola)")

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=20, 
    restore_best_weights=True,
    verbose=1
)


model = create_lstm_model(
    input_shape=(N_STEPS, X.shape[2]),
    units=50,
    output_dim=y.shape[1]
)


print(f"Modelo criado:")
print(f"Input shape: ({N_STEPS}, {X.shape[2]})")
print(f"Output dim: {y.shape[1]}")

model.summary()

# %%
X_test

# %%
EPOCHS = 10
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1

print(f"Configurações:")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Validation split: {VALIDATION_SPLIT}")

print(f"Iniciando treinamento...")
start_time = time.time()


# callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(X_train, y_train,
          validation_split=0.1,
          epochs=10,
          batch_size=1024,
          callbacks=[early_stopping],)



# history = model.fit(
#     X_train, y_train,
#     epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     validation_split=VALIDATION_SPLIT,
#     callbacks=[],
#     verbose=1,
#     shuffle=True, # 
# )

training_time = time.time() - start_time
print(f"\nTreinamento concluído em {training_time:.1f}s")

print(f"\nAvaliação no teste:")
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss: {test_loss:.6f}")
print(f"MAE: {test_mae:.6f}")
print(f"Tempo de treinamento: {training_time:.1f} segundos")


# %% [markdown]
# 

# %%
print(f"\nAvaliação no teste:")
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss: {test_loss:.6f}")
print(f"MAE: {test_mae:.6f}")
print(f"Tempo de treinamento: {training_time:.1f} segundos")

# %%
model_name = f"model_{partida_info['Nome da Partida'].replace(' ', '_')}_steps{N_STEPS}_horizon{PREDICTION_HORIZON}_{time.strftime('%Y%m%d_%H%M%S')}"

import os
if not os.path.exists(f'./models/data/final/{model_name}'):
    os.makedirs(f'./models/data/final/{model_name}')

# %%
with open(f'./models/data/final/{model_name}/model_info.csv', 'w') as f:
    f.write(f"Match Name,{partida_info['Nome da Partida']}\n")
    f.write(f"Input Features,{len(input_features)}\n")
    f.write(f"Output Features,{len(output_features)}\n")
    f.write(f"N_STEPS,{N_STEPS}\n")
    f.write(f"PREDICTION_HORIZON,{PREDICTION_HORIZON}\n")
    f.write(f"EPOCHS,{EPOCHS}\n")
    f.write(f"BATCH_SIZE,{BATCH_SIZE}\n")
    f.write(f"Validation Split,{VALIDATION_SPLIT}\n")
    f.write(f"Test Loss,{test_loss}\n")
    f.write(f"Test MAE,{test_mae}\n")
    f.write(f"Training Time (s),{training_time}\n")

model.save(f'./models/data/{model_name}/{model_name}.h5')


# %%
# # carregar o modelo 
# path_model = '/home/luiz/trabalho-de-conclusao/lstm-ball-prediction/src/notebooks/models/data/model_2024-07-21_07-31_ELIMINATION_PHASE_UBC_Thunderbots-vs-ITAndroids_steps60_horizon1_20251127_094207/model_2024-07-21_07-31_ELIMINATION_PHASE_UBC_Thunderbots-vs-ITAndroids_steps60_horizon1_20251127_094207.h5'
# model_name = path_model.split('/')[-2]

# model_name


# %%
# model = tf.keras.models.load_model(path_model, compile=False)

# import time
# start_time = time.time()
# y_pred = model.predict(X_test, verbose=0)
# end_time = time.time()
# prediction_time = end_time - start_time
# print(f"Tempo de predição para {len(X_test)} amostras: {prediction_time:.4f} segundos")
# print(f"Tempo médio por amostra: {prediction_time / len(X_test):.6f} segundos")
# print(f"Predições shape: {y_pred.shape}")
# print(f"Targets shape: {y_test.shape}")

# # Atualiar o csv com o tempo de predição em model_info.csv
# with open(f'./models/data/{model_name}/model_info.csv', 'a') as f:
#     f.write(f"Prediction Time (s),{prediction_time}\n")

# %%

# calcular o tempo de previsão
import time
start_time = time.time()
y_pred = model.predict(X_test, verbose=0)
end_time = time.time()
prediction_time = end_time - start_time
print(f"Tempo de predição para {len(X_test)} amostras: {prediction_time:.4f} segundos")
print(f"Tempo médio por amostra: {prediction_time / len(X_test):.6f} segundos")
print(f"Predições shape: {y_pred.shape}")
print(f"Targets shape: {y_test.shape}")

# Atualiar o csv com o tempo de predição em model_info.csv
with open(f'./models/data/final/{model_name}/model_info.csv', 'a') as f:
    f.write(f"Prediction Time (s),{prediction_time}\n")


# y_test_denorm = scaler_output.inverse_transform(y_test)    
# y_pred_denorm = scaler_output.inverse_transform(y_pred) 

y_test_denorm = scaler_y.inverse_transform(y_test)    
y_pred_denorm = scaler_y.inverse_transform(y_pred)

# y_test_denorm = y_test
# y_pred_denorm = y_pred

print(f"Predições desnormalizadas: {y_pred_denorm.shape}")

mae_real = np.mean(np.abs(y_pred_denorm - y_test_denorm))
mse_real = np.mean((y_pred_denorm - y_test_denorm)**2)
rmse_real = np.sqrt(mse_real)

print(f"MAE: {mae_real:.2f} mm")
print(f"RMSE: {rmse_real:.2f} mm") 
print(f"MSE: {mse_real:.2f} mm²")

mae_x = np.mean(np.abs(y_pred_denorm[:, 0] - y_test_denorm[:, 0]))
mae_y = np.mean(np.abs(y_pred_denorm[:, 1] - y_test_denorm[:, 1]))

print(f"Erro por coordenada:")
print(f"MAE X: {mae_x:.2f} mm")
print(f"MAE Y: {mae_y:.2f} mm")

with open(f'./models/data/final/{model_name}/error_metrics.csv', 'w') as f:
    f.write(f"MAE,{mae_real}\n")
    f.write(f"RMSE,{rmse_real}\n")
    f.write(f"MSE,{mse_real}\n")
    f.write(f"MAE X,{mae_x}\n")
    f.write(f"MAE Y,{mae_y}\n")

# %%
y_test - y_pred

# %%

def plot_pred_vs_real(ax, real, pred, n_samples, color_points, coord_name):
    """
    Gera scatter real vs predito, linha ideal, pontos ideais e tendência das predições.
    Tudo organizado para fácil edição posterior.
    """

    samples = slice(0, n_samples)

    ax.scatter(real[samples], pred[samples],
               alpha=0.6, color=color_points,
               label=f'Predições (n={n_samples})')

    slope, intercept, r_value, _, _ = stats.linregress(real[samples], pred[samples])
    trend_line = slope * real[samples] + intercept

    ax.plot(real[samples], trend_line, 'r-',
            label=f'Tendência das Predições (R²={r_value**2:.3f})')

    min_val, max_val = real[samples].min(), real[samples].max()
    ax.plot([min_val, max_val], [min_val, max_val],
            'k--', alpha=0.5, label='Predição Ideal (y=x)')

    ax.scatter(real[samples], real[samples],
               color='gray', alpha=0.4, s=20,
               label='Pontos Ideais (y=x)')

    ax.set_xlabel(f'{coord_name} Real (mm)')
    ax.set_ylabel(f'{coord_name} Predito (mm)')
    ax.set_title(f'Predição vs Real - Coordenada {coord_name}')
    ax.grid(True, alpha=0.3)
    ax.legend()

    return slope, r_value**2


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
n_samples = 50

slope_x, r2_x = plot_pred_vs_real(
    ax1,
    y_test_denorm[:, 0],
    y_pred_denorm[:, 0],
    n_samples,
    color_points='blue',
    coord_name='X'
)

slope_y, r2_y = plot_pred_vs_real(
    ax2,
    y_test_denorm[:, 1],
    y_pred_denorm[:, 1],
    n_samples,
    color_points='green',
    coord_name='Y'
)

plt.tight_layout()
plt.show()

print("ANÁLISE DE CORRELAÇÃO:")
print(f"Coordenada X: R² = {r2_x:.3f}, Slope = {slope_x:.3f}")
print(f"Coordenada Y: R² = {r2_y:.3f}, Slope = {slope_y:.3f}")
print(f"Erro médio: {mae_real:.1f} mm")


# %%
erro_x = np.abs(y_pred_denorm[:, 0] - y_test_denorm[:, 0])

erro_y = np.abs(y_pred_denorm[:, 1] - y_test_denorm[:, 1])

plt.figure(figsize=(14,6))

plt.plot(erro_x, label='Erro |Pred - Real| (X)', linewidth=1)
plt.plot(erro_y, label='Erro |Pred - Real| (Y)', linewidth=1)

plt.title("Erro Absoluto por Amostra")
plt.xlabel("Índice da Amostra")
plt.ylabel("Erro (mm)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(f"./models/data/final/{model_name}/erro_absoluto.png")
plt.show()

print("ERRO MÉDIO X:", np.mean(erro_x))
print("ERRO MÉDIO Y:", np.mean(erro_y))
print("ERRO MÁX X:", np.max(erro_x))
print("ERRO MÁX Y:", np.max(erro_y))



# plotantdo trajetoria prevista vs real no plt.plot
plt.figure(figsize=(14,6))
plt.plot(y_test_denorm[:, 0], y_test_denorm[:, 1], label='Trajetória Real', color='blue', linewidth=2)
plt.plot(y_pred_denorm[:, 0], y_pred_denorm[:, 1], label='Trajetória Predita', color='red', linewidth=2)
plt.title("Trajetória: Real vs Predita")
plt.xlabel("Posição X (mm)")
plt.ylabel("Posição Y (mm)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(f"./models/data/final/{model_name}/trajetoria_real_vs_predita.png")
plt.show()


# Distribuição dos erros de predição
erro_magnitude = np.sqrt(erro_x**2 + erro_y**2)
plt.figure(figsize=(14,6))
plt.hist(erro_magnitude, bins=50, color='purple', alpha=0.7)
plt.title("Distribuição dos Erros de Predição (Magnitude)")
plt.xlabel("Erro (mm)")
plt.ylabel("Frequência")
plt.grid(True, alpha=0.3)
plt.savefig(f"./models/data/final/{model_name}/distribuicao_erro_predicao.png")
plt.show()

# Distribuição dos erros de predição X e Y
plt.figure(figsize=(14,6))
plt.subplot(1, 2, 1)
plt.hist(erro_x, bins=50, color='orange', alpha=0.7)
plt.title("Distribuição dos Erros de Predição (X)")
plt.xlabel("Erro X (mm)")
plt.ylabel("Frequência")
plt.grid(True, alpha=0.3)
plt.savefig(f"./models/data/final/{model_name}/distribuicao_erro_predicao_x.png")
plt.show()

plt.figure(figsize=(14,6))
plt.subplot(1, 2, 2)
plt.hist(erro_y, bins=50, color='green', alpha=0.7)
plt.title("Distribuição dos Erros de Predição (Y)")
plt.xlabel("Erro Y (mm)")
plt.ylabel("Frequência")
plt.grid(True, alpha=0.3)
plt.savefig(f"./models/data/final/{model_name}/distribuicao_erro_predicao_y.png")
plt.show()

# %%
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.savefig(f"./models/data/final/{model_name}/loss_mae.png")
plt.show()

# %%
start_idx = 35000
seq_length = 12000

X_sequential = []
y_sequential = []

# for i in range(seq_length):
#     if start_idx + N_STEPS + i < len(data_input_normalized):
#         seq_x = data_input_normalized[start_idx + i : start_idx + i + N_STEPS]
#         seq_y = data_output_normalized[start_idx + i + N_STEPS]
#         X_sequential.append(seq_x)
#         y_sequential.append(seq_y)

# usar a função de criação de sequências
X, y = create_sequences_custom_fixed(
    input_data=data_input_normalized[start_idx: start_idx + seq_length + N_STEPS],
    output_data=data_output_normalized[start_idx: start_idx + seq_length + N_STEPS],
    n_steps=N_STEPS,
    max_samples=seq_length,
    prediction_horizon=0
)
X_sequential = X
y_sequential = y


# X_sequential = np.array(X_sequential)
# y_sequential = np.array(y_sequential)



print(f"Seq: X={X_sequential.shape}, y={y_sequential.shape}")

#distribuição do tempo de predição
time_start = time.time()
y_pred_sequential = model.predict(X_sequential, verbose=0)
time_end = time.time()
print(f"Prediction time: {time_end - time_start} seconds")

y_true_seq = scaler_y.inverse_transform(y_sequential)
y_pred_seq = scaler_y.inverse_transform(y_pred_sequential)

fig, ax = drawField()
plot_trajectory(ax, y_true_seq[:, 0], y_true_seq[:, 1], label='Real (Sequencial)', color='green')
plot_trajectory(ax, y_pred_seq[:, 0], y_pred_seq[:, 1], label='Predito (Sequencial)', color='orange')
plt.title('Trajetória Sequencial Real vs Predita (LSTM)')
plt.legend()
plt.savefig(f"./models/data/final/{model_name}/trajetoria_sequencial.png")
plt.show()

plt.figure(figsize=(10,6))
plt.plot(y_true_seq[:,0], y_true_seq[:,1], label='Real (Sequencial)', color='green')
plt.plot(y_pred_seq[:,0], y_pred_seq[:,1], label='Predito (LSTM - Sequencial)', color='orange')
plt.legend()
plt.title('Trajetória Sequencial Real vs Predita LSTM')
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.savefig(f"./models/data/final/{model_name}/trajetoria_sequencial_xy.png")
plt.show()

print(y_true_seq.shape, y_pred_seq.shape)

mae_seq = np.mean(np.abs(y_pred_seq - y_true_seq), axis=1)
rmse_seq = np.sqrt(np.mean((y_pred_seq - y_true_seq)**2, axis=1))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(mae_seq, label='MAE ao longo da sequência', color='blue')
plt.title('MAE ao longo da sequência')
plt.xlabel('Índice da Amostra na Sequência')
plt.ylabel('MAE (mm)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"./models/data/final/{model_name}/mae_sequencia.png")

plt.subplot(1, 2, 2)
plt.plot(rmse_seq, label='RMSE ao longo da sequência', color='red')

plt.title('RMSE ao longo da sequência')
plt.xlabel('Índice da Amostra na Sequência')
plt.ylabel('RMSE (mm)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"./models/data/final/{model_name}/mae_rmse_sequencia.png")

# erro no eixo x, y no mesmo gráfico
plt.figure(figsize=(10, 6))
plt.plot(np.abs(y_pred_seq[:, 0] - y_true_seq[:, 0]), label='Erro Absoluto X', color='blue')
plt.plot(np.abs(y_pred_seq[:, 1] - y_true_seq[:, 1]), label='Erro Absoluto Y', color='red')
plt.title('Erro Absoluto por Coordenada na Sequência')
plt.xlabel('Índice da Amostra na Sequência')
plt.ylabel('Erro Absoluto (mm)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(f"./models/data/final/{model_name}/erro_absoluto_sequencia.png")
plt.show()

#distribuição dos erros
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(np.abs(y_pred_seq[:, 0] - y_true_seq[:, 0]), bins=30, color='blue', alpha=0.7)
plt.title('Distribuição do Erro Absoluto X')
plt.xlabel('Erro Absoluto X (mm)')
plt.ylabel('Frequência')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"./models/data/final/{model_name}/distribuicao_erro_absoluto_x.png")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(np.abs(y_pred_seq[:, 1] - y_true_seq[:, 1]), bins=30, color='red', alpha=0.7)
plt.title('Distribuição do Erro Absoluto Y')
plt.xlabel('Erro Absoluto Y (mm)')
plt.ylabel('Frequência')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("MAE em X na sequência:", np.mean(np.abs(y_pred_seq[:, 0] - y_true_seq[:, 0])))
print("MAE em Y na sequência:", np.mean(np.abs(y_pred_seq[:, 1] - y_true_seq[:, 1])))

# erro quadrático médio na sequência
mse_x_seq = np.mean((y_pred_seq[:, 0] - y_true_seq[:, 0])**2)
mse_y_seq = np.mean((y_pred_seq[:, 1] - y_true_seq[:, 1])**2)
print("MSE em X na sequência:", mse_x_seq)
print("MSE em Y na sequência:", mse_y_seq)

rmse_x_seq = np.sqrt(mse_x_seq)
rmse_y_seq = np.sqrt(mse_y_seq)
print("RMSE em X na sequência:", rmse_x_seq)
print("RMSE em Y na sequência:", rmse_y_seq)

distances_real = np.sqrt(np.diff(y_true_seq[:, 0])**2 + np.diff(y_true_seq[:, 1])**2)
distances_pred = np.sqrt(np.diff(y_pred_seq[:, 0])**2 + np.diff(y_pred_seq[:, 1])**2)

# %% [markdown]
# 


