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
            
            bpfo = (n_bolas / 2) * fr * (1 - beta)
            bpfi = (n_bolas / 2) * fr * (1 + beta)
            bsf = (pd_val / b['d_bola']) * fr * (1 - beta**2)
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
        
        tk.Button(self.side_bar, text="EXECUTAR DIAGNÓSTICO ML", command=self.run_ml_diagnostic, bg="#27ae60", fg="white", font=("Arial", 12, "bold")).pack(fill=tk.X, pady=30)

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
                                         "3. Um pico de amplitude conhecida (clique próximo ao topo)")

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
        """Extração robusta do envelope superior do espectro"""
        p1, p2, p3 = self.calib_points
        
        # Margem maior para garantir captura de barras nas bordas e topo
        margin_x = 50
        margin_y = 60
        
        # Limites baseados APENAS em P1 e P2 para X (extremos do gráfico)
        min_x_base = min(p1[0], p2[0])
        max_x_base = max(p1[0], p2[0])
        
        # Crop inclui margem, mas depois filtraremos apenas a região real do gráfico
        x_start = int(min_x_base - margin_x)
        x_end = int(max_x_base + margin_x)
        y_top = int(min(p1[1], p2[1], p3[1]) - margin_y)
        y_base = int(max(p1[1], p2[1], p3[1]) + margin_y)
        
        x_start = max(0, x_start)
        x_end = min(self.image_np.shape[1], x_end)
        y_top = max(0, y_top)
        y_base = min(self.image_np.shape[0], y_base)
        
        crop = self.image_np[y_top:y_base, x_start:x_end]
        if crop.size == 0:
            messagebox.showerror("Erro", "Região de crop inválida.")
            return
        
        thresh = np.mean(crop) - 2.5 * np.std(crop)
        min_thickness = 5
        
        self.points = []
        for col in range(crop.shape[1]):
            column = crop[:, col]
            dark_idx = np.where(column < thresh)[0]
            
            if len(dark_idx) >= min_thickness:
                top_y = np.min(dark_idx)
            else:
                top_y = crop.shape[0] - 1
            
            global_y = y_top + top_y
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
            
        p1, p2, p3 = self.calib_points
        
        # Limites STRICT do eixo X (definidos apenas por P1 e P2 - extremos do gráfico)
        min_x_pixel = min(p1[0], p2[0])
        max_x_pixel = max(p1[0], p2[0])
        
        # Verifica se o pico calibrado está dentro dos limites do gráfico
        if not (min_x_pixel <= p3[0] <= max_x_pixel):
            messagebox.showwarning("Aviso", "O ponto de pico (P3) está fora dos limites do eixo X definido por P1 e P2.\n"
                                           "Recomenda-se clicar em um pico dentro do gráfico.")
        
        base_y_pixel = (p1[1] + p2[1]) / 2.0
        
        dx_pixel = p2[0] - p1[0]
        if abs(dx_pixel) < 10:
            messagebox.showerror("Erro", "Pontos X muito próximos.")
            return
        scale_x = (p2[2] - p1[2]) / dx_pixel
        
        # Calibração Y usando pico mais alto na janela (apenas dentro dos limites do gráfico)
        window_radius = 80
        nearby_points = [pt for pt in self.points 
                         if abs(pt[0] - p3[0]) < window_radius and min_x_pixel <= pt[0] <= max_x_pixel]
        
        if not nearby_points:
            messagebox.showerror("Erro", "Nenhum ponto detectado próximo ao pico calibrado dentro do gráfico.")
            return
        
        peak_digit_py = min(py for _, py in nearby_points)
        
        delta_pixel_y = base_y_pixel - peak_digit_py
        if delta_pixel_y <= 20:
            messagebox.showerror("Erro", "Pico calibrado detectado muito baixo.")
            return
        
        scale_y = p3[2] / delta_pixel_y
        
        self.real_data = []
        for px, py in self.points:
            # FILTRA RIGOROSAMENTE: só pontos dentro dos extremos P1-P2 (ignora tudo fora do gráfico)
            if min_x_pixel <= px <= max_x_pixel:
                rx = p1[2] + (px - p1[0]) * scale_x
                ry = (base_y_pixel - py) * scale_y
                ry = max(0.0, ry)
                if rx >= 0:
                    self.real_data.append((rx, ry))
        
        self.real_data.sort(key=lambda x: x[0])
        
        # Marca o pico usado para calibração Y
        self.ax_img.plot(p3[0], peak_digit_py, 'co', markersize=12, alpha=0.8)
        
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
            self.ax_spec.plot(freqs, amps, color='blue', linewidth=1.5)
            self.ax_spec.grid(True, alpha=0.3)
            self.ax_spec.set_title("Espectro Digitalizado", fontsize=14)
            self.ax_spec.set_xlabel("Frequência (Hz)")
            self.ax_spec.set_ylabel("Amplitude (g)")
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
            'BPFO': (b['n_bolas']/2) * fr * (1 - beta),
            'BPFI': (b['n_bolas']/2) * fr * (1 + beta),
            'BSF':  (pd_val / b['d_bola']) * fr * (1 - beta**2),
            'FTF':  0.5 * fr * (1 - beta)
        }
        
        self.update_plots()
        freqs, amps = zip(*self.real_data)
        freqs = np.array(freqs)
        amps = np.array(amps)
        
        for name, f in f_targets.items():
            if min(freqs) <= f <= max(freqs):
                self.ax_spec.axvline(f, color='red', linestyle='--', linewidth=2, alpha=0.8)
                self.ax_spec.text(f, max(amps)*0.95, name, color='red', fontsize=10, rotation=90, va='top', ha='left')
        
        peaks, _ = find_peaks(amps, prominence=0.05*np.ptp(amps), distance=len(amps)//50)
        self.ax_spec.plot(freqs[peaks], amps[peaks], "o", color='magenta', markersize=6)
        
        self.canvas.draw()
        
        rms = np.sqrt(np.mean(amps**2))
        peak = np.max(amps)
        cf = peak / rms if rms > 0 else 0
        kurt = kurtosis(amps) + 3
        
        amps_at_f = []
        for ft in f_targets.values():
            idx = np.argmin(np.abs(freqs - ft))
            amps_at_f.append(amps[idx])
        
        features = [self.rpm, age, rms] + amps_at_f + [kurt, peak, cf] + list(f_targets.values())
        
        results = self.predictor.predict(features)
        self.show_dashboard(results)

    def show_dashboard(self, probs):
        dash = tk.Toplevel(self.root)
        dash.title("Diagnóstico Inteligente")
        dash.geometry("700x600")
        
        fig_dash = Figure(figsize=(7, 5))
        ax = fig_dash.add_subplot(111)
        
        labels = list(probs.keys())
        values = [probs[l] * 100 for l in labels]
        colors = ['#2ecc71' if l == 'Normal' else '#e74c3c' for l in labels]
        
        ax.barh(labels, values, color=colors)
        ax.set_xlim(0, 100)
        ax.set_title(f"Probabilidades de Falha - {self.bearing_model} @ {self.rpm} RPM", fontsize=14)
        ax.set_xlabel("Confiança (%)")
        for i, v in enumerate(values):
            ax.text(v + 2, i, f"{v:.1f}%", va='center', fontsize=12)
        
        canvas_dash = FigureCanvasTkAgg(fig_dash, master=dash)
        canvas_dash.draw()
        canvas_dash.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        veredito = max(probs, key=probs.get)
        cor = "green" if veredito == 'Normal' else "red"
        tk.Label(dash, text=f"VEREDITO: {veredito}", font=("Arial", 18, "bold"), fg=cor).pack(pady=20)

if __name__ == "__main__":
    root = tk.Tk()
    app = IntegratedVibrationApp(root)
    root.mainloop()