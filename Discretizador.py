import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image
import os
import joblib
from scipy.signal import find_peaks
from scipy.stats import kurtosis
import random
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- Configurações de Rolamentos (valores precisos) ---
BEARING_DB = {
    "6203ZZ": {"d_interno": 17, "d_externo": 40, "d_bola": 6.747, "n_bolas": 8},
    "6204ZZ": {"d_interno": 20, "d_externo": 47, "d_bola": 7.938, "n_bolas": 9}
}

# --- Módulo de Machine Learning ---
class FaultPredictor:
    def __init__(self):
        self.model_file = "bearing_predictor_gb.pkl"
        self.scaler_file = "bearing_scaler.pkl"
        self.model = None
        self.scaler = None
        self.classes = ['Normal', 'BPFO', 'BPFI', 'BSF', 'FTF']
        self.init_model()

    def init_model(self):
        if os.path.exists(self.model_file) and os.path.exists(self.scaler_file):
            try:
                self.model = joblib.load(self.model_file)
                self.scaler = joblib.load(self.scaler_file)
            except:
                self.train_model()
        else:
            self.train_model()

    def generate_synthetic_data(self, n_samples=5000):
        np.random.seed(42)
        X, y = [], []
        bearings = list(BEARING_DB.values())
        
        for _ in range(n_samples):
            rpm = np.random.uniform(500, 3000)
            fr = rpm / 60
            age = np.random.uniform(0, 5000)
            
            b = random.choice(bearings)
            pd_val = (b['d_interno'] + b['d_externo']) / 2
            beta = b['d_bola'] / pd_val
            n_bolas = b['n_bolas']
            
            bpfo = (n_bolas / 2) * fr * (1 + beta)
            bpfi = (n_bolas / 2) * fr * (1 - beta)
            bsf = (pd_val / (2 * b['d_bola'])) * fr * (1 - beta**2)
            ftf = 0.5 * fr * (1 - beta)
            f_vals = [bpfo, bpfi, bsf, ftf]
            
            base_vib = np.random.normal(0.05, 0.01)
            amps = {cls: (np.random.uniform(0.1, 0.6) if np.random.rand() < 0.25 else 0) 
                    for cls in self.classes[1:]}
            
            total_fault_amp = sum(amps.values())
            rms = base_vib + total_fault_amp
            kurt = 3 + 12 * total_fault_amp
            peak = rms * (3 + 8 * total_fault_amp)
            cf = peak / rms if rms > 0 else 3.0
            
            feat = [rpm, age, rms] + list(amps.values()) + [kurt, peak, cf] + f_vals
            X.append(feat)
            y.append('Normal' if total_fault_amp < 0.05 else max(amps, key=amps.get))
            
        return np.array(X), np.array(y)

    def train_model(self):
        X, y = self.generate_synthetic_data()
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5))
        ])
        pipeline.fit(X, y)
        self.model = pipeline.named_steps['clf']
        self.scaler = pipeline.named_steps['scaler']
        joblib.dump(self.model, self.model_file)
        joblib.dump(self.scaler, self.scaler_file)

    def predict(self, features):
        try:
            fs = self.scaler.transform([features])
            probs = self.model.predict_proba(fs)[0]
            return {cls: prob for cls, prob in zip(self.classes, probs)}
        except:
            return {cls: 0.2 for cls in self.classes}

# --- Aplicação Principal ---
class IntegratedVibrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("UFSJ-CAP: Diagnóstico Avançado de Rolamentos")
        self.root.geometry("1600x900")
        
        self.predictor = FaultPredictor()
        self.image_raw = None
        self.image_np = None
        self.points = []
        self.calib_points = []
        self.real_data = []
        self.rpm = 1200
        self.bearing_model = "6203ZZ"

        self.setup_ui()

    def setup_ui(self):
        self.side_bar = tk.Frame(self.root, width=320, bg="#2c3e50", padx=15, pady=15)
        self.side_bar.pack(side=tk.LEFT, fill=tk.Y)
        
        tk.Label(self.side_bar, text="COMANDOS", fg="white", bg="#2c3e50", font=("Arial", 14, "bold")).pack(pady=15)
        
        tk.Button(self.side_bar, text="1. Carregar Foto do Gráfico", command=self.load_image, bg="#3498db", fg="white", font=("Arial", 10, "bold")).pack(fill=tk.X, pady=8)
        self.btn_calib = tk.Button(self.side_bar, text="2. Calibrar Eixos", command=self.start_calibration, state=tk.DISABLED, bg="#f1c40f", fg="black")
        self.btn_calib.pack(fill=tk.X, pady=8)
        
        tk.Button(self.side_bar, text="Alternativa: Carregar CSV", command=self.load_csv, bg="#95a5a6", fg="white").pack(fill=tk.X, pady=20)
        
        tk.Label(self.side_bar, text="RPM do Eixo:", fg="white", bg="#2c3e50", font=("Arial", 10)).pack(anchor='w')
        self.ent_rpm = tk.Entry(self.side_bar)
        self.ent_rpm.insert(0, "1200")
        self.ent_rpm.pack(fill=tk.X, pady=5)
        
        tk.Label(self.side_bar, text="Modelo do Rolamento:", fg="white", bg="#2c3e50", font=("Arial", 10)).pack(anchor='w')
        self.combo_bearing = ttk.Combobox(self.side_bar, values=list(BEARING_DB.keys()), state="readonly")
        self.combo_bearing.set("6203ZZ")
        self.combo_bearing.pack(fill=tk.X, pady=5)
        
        tk.Label(self.side_bar, text="Idade do Rolamento (horas):", fg="white", bg="#2c3e50", font=("Arial", 10)).pack(anchor='w')
        self.ent_age = tk.Entry(self.side_bar)
        self.ent_age.insert(0, "2500")
        self.ent_age.pack(fill=tk.X, pady=5)
        
        tk.Button(self.side_bar, text="EXECUTAR DIAGNÓSTICO", command=self.run_ml_diagnostic, bg="#27ae60", fg="white", font=("Arial", 12, "bold")).pack(fill=tk.X, pady=30)

        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.ax_img = self.fig.add_subplot(211)
        self.ax_spec = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.image_raw = Image.open(path).convert('RGB')
            self.image_np = np.array(self.image_raw.convert('L'))
            self.ax_img.clear()
            self.ax_img.imshow(self.image_raw)
            self.ax_img.axis('off')
            self.ax_img.set_title("Clique para calibrar: 1. Origem → 2. Fim eixo X → 3. Pico conhecido", fontsize=12)
            self.canvas.draw()
            self.btn_calib['state'] = tk.NORMAL
            self.calib_points = []
            self.points = []
            self.real_data = []

    def start_calibration(self):
        messagebox.showinfo("Calibração", "Clique em 3 pontos na imagem:\n\n"
                                         "1. Origem (canto inferior esquerdo do gráfico)\n"
                                         "2. Final do eixo X (mesma altura da origem)\n"
                                         "3. Pico de amplitude conhecida (PREFERENCIALMENTE O MAIS ALTO)")

    def on_canvas_click(self, event):
        if event.inaxes != self.ax_img or self.image_np is None:
            return
        
        if len(self.calib_points) >= 3:
            return
        
        prompts = [
            "Valor de frequência na origem (geralmente 0 Hz):",
            "Valor de frequência no final do eixo X (Hz):",
            "Valor de amplitude conhecida neste pico (g):"
        ]
        
        val = simpledialog.askfloat(f"Ponto {len(self.calib_points)+1}", prompts[len(self.calib_points)], minvalue=0)
        if val is None:
            return
        
        self.calib_points.append((event.xdata, event.ydata, val))
        self.ax_img.plot(event.xdata, event.ydata, 'ro', markersize=10)
        self.canvas.draw()
        
        if len(self.calib_points) == 3:
            dy_base = abs(self.calib_points[0][1] - self.calib_points[1][1])
            if dy_base > self.image_np.shape[0] * 0.05:
                messagebox.showwarning("Aviso", f"Baseline com diferença de {dy_base:.0f} pixels. "
                                             "Calibração pode ter pequena imprecisão.")
            self.digitize_image()

    def digitize_image(self):
        """Extração do envelope superior usando o pixel mais escuro por coluna"""
        p1_raw, p2_raw, p3 = self.calib_points
        
        margin_x = 50
        margin_top = 30
        margin_bottom = 80
        
        min_x_base = min(p1_raw[0], p2_raw[0])
        max_x_base = max(p1_raw[0], p2_raw[0])
        
        x_start = max(0, int(min_x_base - margin_x))
        x_end = min(self.image_np.shape[1], int(max_x_base + margin_x))
        
        y_top = max(0, int(min(p1_raw[1], p2_raw[1], p3[1]) - margin_top))
        y_base = min(self.image_np.shape[0], int(max(p1_raw[1], p2_raw[1], p3[1]) + margin_bottom))
        
        crop = self.image_np[y_top:y_base, x_start:x_end]
        if crop.size == 0:
            messagebox.showerror("Erro", "Região de crop inválida.")
            return
        
        self.points = []
        for col in range(crop.shape[1]):
            column = crop[:, col]
            top_y_local = np.argmin(column)
            global_y = y_top + top_y_local
            global_x = x_start + col
            self.points.append((global_x, global_y))
        
        if self.points:
            xs, ys = zip(*self.points)
            self.ax_img.plot(xs, ys, color='lime', linewidth=2.5, alpha=0.9)
            self.canvas.draw()
        
        self.convert_to_real_units()

    def convert_to_real_units(self):
        if not self.points or len(self.calib_points) < 3:
            return
            
        base_points_raw = self.calib_points[:2]
        p3 = self.calib_points[2]
        
        base_points_raw.sort(key=lambda pt: pt[0])
        left_p = base_points_raw[0]
        right_p = base_points_raw[1]
        
        min_x_pixel = left_p[0]
        max_x_pixel = right_p[0]
        
        if abs(right_p[0] - left_p[0]) < 10:
            messagebox.showerror("Erro", "Pontos base X muito próximos.")
            return
            
        scale_x = (right_p[2] - left_p[2]) / (right_p[0] - left_p[0])
        base_y_pixel = (left_p[1] + right_p[1]) / 2.0
        
        window_radius = 120
        nearby_points = [pt for pt in self.points 
                         if abs(pt[0] - p3[0]) < window_radius and min_x_pixel <= pt[0] <= max_x_pixel]
        
        if not nearby_points:
            messagebox.showerror("Erro", "Nenhum ponto detectado próximo ao pico calibrado.")
            return
        
        peak_digit_py = min(py for _, py in nearby_points)
        
        delta_pixel_y = base_y_pixel - peak_digit_py
        if delta_pixel_y <= 30:
            messagebox.showerror("Erro", "Pico detectado muito baixo. Clique exatamente no topo do pico desejado.")
            return
        
        scale_y = p3[2] / delta_pixel_y
        
        self.ax_img.plot(p3[0], peak_digit_py, 'co', markersize=12, alpha=0.8)
        self.canvas.draw()
        
        self.real_data = []
        max_allowed_amp = p3[2]
        for px, py in self.points:
            if min_x_pixel <= px <= max_x_pixel:
                rx = left_p[2] + (px - left_p[0]) * scale_x
                ry = (base_y_pixel - py) * scale_y
                ry = min(ry, max_allowed_amp)
                ry = max(0.0, ry)
                if rx >= 0:
                    self.real_data.append((rx, ry))
        
        self.real_data.sort(key=lambda x: x[0])
        self.update_plots()

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if path:
            df = pd.read_csv(path, header=None)
            self.real_data = [(float(row[0]), float(row[1])) for _, row in df.iterrows()]
            self.real_data.sort(key=lambda x: x[0])
            self.update_plots()

    def update_plots(self):
        self.ax_spec.clear()
        if self.real_data:
            freqs, amps = zip(*self.real_data)
            freqs = np.array(freqs)
            amps = np.array(amps)
            
            self.ax_spec.plot(freqs, amps, color='blue', linewidth=1.5, label='Espectro')
            self.ax_spec.grid(True, alpha=0.3)
            
            # Calcular frequências de defeito baseado no RPM atual
            try:
                rpm = float(self.ent_rpm.get())
                fr = rpm / 60
                b = BEARING_DB[self.combo_bearing.get()]
                pd_val = (b['d_interno'] + b['d_externo']) / 2
                beta = b['d_bola'] / pd_val
                
                f_targets = {
                    'BPFO': (b['n_bolas']/2) * fr * (1 + beta),
                    'BPFI': (b['n_bolas']/2) * fr * (1 - beta),
                    'BSF':  (pd_val / (2 * b['d_bola'])) * fr * (1 - beta**2),
                    'FTF':  0.5 * fr * (1 - beta)
                }
                
                # Cores para cada tipo de defeito
                colors = {'BPFO': '#FF6B6B', 'BPFI': '#4ECDC4', 'BSF': '#45B7D1', 'FTF': '#FFA07A'}
                
                # Plotar harmônicos de cada defeito
                for fault_name, base_freq in f_targets.items():
                    if base_freq <= 0:
                        continue
                    
                    # Plotar até 8 harmônicos
                    for h in range(1, 9):
                        harmonic_freq = h * base_freq
                        
                        # Plotar linha vertical do harmônico
                        if 0 <= harmonic_freq <= max(freqs):
                            self.ax_spec.axvline(harmonic_freq, color=colors[fault_name], 
                                                linestyle='--', linewidth=1.5, alpha=0.7)
                            
                            # Adicionar label em CADA harmônico
                            label_text = f"{fault_name}\n{h}x"
                            self.ax_spec.text(harmonic_freq, max(amps)*0.92, label_text, 
                                            color=colors[fault_name], fontsize=8, 
                                            rotation=90, va='top', ha='right', fontweight='bold',
                                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            except:
                pass
            
            self.ax_spec.set_title("Espectro Digitalizado com Harmônicos de Defeitos", fontsize=14)
            self.ax_spec.set_xlabel("Frequência (Hz)")
            self.ax_spec.set_ylabel("Amplitude (g)")
            self.ax_spec.legend(loc='upper right')
        else:
            self.ax_spec.set_title("Espectro Digitalizado (vazio)")
        
        self.canvas.draw()

    def run_ml_diagnostic(self):
        if not self.real_data:
            messagebox.showerror("Erro", "Carregue uma imagem/CSV e digitalize primeiro!")
            return
        
        try:
            self.rpm = float(self.ent_rpm.get())
            age = float(self.ent_age.get())
        except:
            messagebox.showerror("Erro", "RPM ou idade inválidos!")
            return
        
        self.bearing_model = self.combo_bearing.get()
        fr = self.rpm / 60
        b = BEARING_DB[self.bearing_model]
        pd_val = (b['d_interno'] + b['d_externo']) / 2
        beta = b['d_bola'] / pd_val
        
        f_targets = {
            'BPFO': (b['n_bolas']/2) * fr * (1 + beta),
            'BPFI': (b['n_bolas']/2) * fr * (1 - beta),
            'BSF':  (pd_val / (2 * b['d_bola'])) * fr * (1 - beta**2),
            'FTF':  0.5 * fr * (1 - beta)
        }
        
        self.update_plots()
        freqs, amps = zip(*self.real_data)
        freqs = np.array(freqs)
        amps = np.array(amps)
        
        peaks, _ = find_peaks(amps, prominence=0.05*np.ptp(amps), distance=len(amps)//50)
        self.ax_spec.plot(freqs[peaks], amps[peaks], "o", color='magenta', markersize=6)
        self.canvas.draw()
        
        # DETECÇÃO DE DEFEITOS POR HARMÔNICOS
        tolerance = 0.05
        fault_scores = {'BPFO': 0, 'BPFI': 0, 'BSF': 0, 'FTF': 0}
        
        for fault_name, base_freq in f_targets.items():
            if base_freq <= 0:
                continue
            
            harmonic_count = 0
            
            for h in range(1, 6):
                target_freq = h * base_freq
                if target_freq > max(freqs):
                    break
                
                matches = [i for i in range(len(peaks)) if abs(freqs[peaks[i]] - target_freq) < tolerance * target_freq]
                
                if matches:
                    harmonic_count += 1
            
            if harmonic_count > 0:
                fault_scores[fault_name] = (harmonic_count / 5.0) * 100
        
        max_fault = max(fault_scores, key=fault_scores.get)
        max_score = fault_scores[max_fault]
        
        if max_score < 20:
            veredito = "Normal"
            cor = "green"
        else:
            veredito = max_fault
            cor = "red"
        
        self.show_dashboard_simple(veredito, cor, fault_scores)

    def show_dashboard_simple(self, veredito, cor, scores):
        dash = tk.Toplevel(self.root)
        dash.title("Diagnóstico")
        dash.geometry("700x500")
        
        fig_dash = Figure(figsize=(7, 5))
        ax = fig_dash.add_subplot(111)
        
        labels = list(scores.keys())
        values = list(scores.values())
        colors = ['#2ecc71' if v < 20 else '#e74c3c' for v in values]
        
        ax.barh(labels, values, color=colors)
        ax.set_xlim(0, 100)
        ax.set_title(f"Detecção de Defeitos - {self.bearing_model} @ {self.rpm} RPM", fontsize=14)
        ax.set_xlabel("Score de Detecção (%)")
        for i, v in enumerate(values):
            ax.text(v + 2, i, f"{v:.1f}%", va='center', fontsize=12)
        
        canvas_dash = FigureCanvasTkAgg(fig_dash, master=dash)
        canvas_dash.draw()
        canvas_dash.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(dash, text=f"VEREDITO: {veredito}", font=("Arial", 18, "bold"), fg=cor).pack(pady=20)

if __name__ == "__main__":
    root = tk.Tk()
    app = IntegratedVibrationApp(root)
    root.mainloop()