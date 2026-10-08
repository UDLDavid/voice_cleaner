#!/usr/bin/env python3
"""Reduce el ruido de fondo de un video y realza la voz humana.

Flujo:
  1. ffmpeg extrae el audio a WAV.
  2. Filtro pasa-altos (quita retumbe) + reducción de ruido espectral (noisereduce).
  3. ffmpeg aplica ecualización de voz, compresión y normalización de volumen,
     y vuelve a unir el audio limpio con el video original (sin recodificar el video).

Uso:
  python limpiar_voz.py                      (abre la ventana)
  python limpiar_voz.py entrada.mp4
  python limpiar_voz.py entrada.mp4 -o salida.mp4 --fuerza 0.9
  python limpiar_voz.py entrada.mp4 --ruido-variable
"""

import argparse
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import noisereduce as nr
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

SAMPLE_RATE = 48000
EMPAQUETADO = getattr(sys, "frozen", False)  # True dentro del .exe de PyInstaller


class ErrorProceso(Exception):
    pass


def herramienta(nombre):
    """Busca ffmpeg/ffprobe: primero junto al .exe (empaquetado), luego en el PATH."""
    exe = nombre + (".exe" if os.name == "nt" else "")
    if EMPAQUETADO:
        for carpeta in (Path(getattr(sys, "_MEIPASS", "")), Path(sys.executable).parent):
            if (carpeta / exe).is_file():
                return str(carpeta / exe)
    encontrado = shutil.which(nombre)
    if not encontrado:
        raise ErrorProceso(f"No se encontró {nombre}. Instala ffmpeg o colócalo junto al programa.")
    return encontrado


def run(cmd):
    # En Windows evita que se abra una consola negra por cada llamada a ffmpeg.
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", creationflags=flags)
    if result.returncode != 0:
        raise ErrorProceso(f"Error ejecutando {Path(cmd[0]).stem}:\n{result.stderr[-2000:]}")
    return result.stdout


def tiene_audio(video):
    out = run([herramienta("ffprobe"), "-v", "error", "-select_streams", "a",
               "-show_entries", "stream=index", "-of", "csv=p=0", str(video)])
    return bool(out.strip())


def extraer_audio(video, wav):
    run([herramienta("ffmpeg"), "-y", "-v", "error", "-i", str(video), "-vn",
         "-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le", str(wav)])


def limpiar(wav_in, wav_out, fuerza, variable):
    audio, sr = sf.read(wav_in, dtype="float32", always_2d=True)  # (muestras, canales)

    # Pasa-altos a 80 Hz: elimina zumbidos graves, viento y golpes que no son voz.
    sos = butter(4, 80, btype="highpass", fs=sr, output="sos")
    audio = sosfiltfilt(sos, audio, axis=0).astype(np.float32)

    # Reducción de ruido espectral. Modo estacionario: perfil de ruido fijo
    # (ventiladores, aire acondicionado, zumbido, siseo); modo variable: el perfil
    # se re-estima continuamente (tráfico, gente hablando de fondo).
    limpio = nr.reduce_noise(
        y=audio.T,
        sr=sr,
        stationary=not variable,
        prop_decrease=fuerza,
        n_jobs=1 if EMPAQUETADO else -1,  # multiproceso de joblib no es fiable dentro del .exe
    ).T

    sf.write(wav_out, limpio, sr, subtype="FLOAT")


def unir(video, wav, salida):
    filtros = ",".join([
        "equalizer=f=250:t=q:w=1:g=-2",     # menos "barro" en graves-medios
        "equalizer=f=3000:t=q:w=1.2:g=4",   # presencia / inteligibilidad de la voz
        "lowpass=f=12000",                  # recorta siseo agudo residual
        "acompressor=threshold=-20dB:ratio=3:attack=5:release=100:makeup=2",
        "loudnorm=I=-16:TP=-1.5:LRA=11",    # volumen final uniforme
    ])
    run([herramienta("ffmpeg"), "-y", "-v", "error",
         "-i", str(video), "-i", str(wav),
         "-map", "0:v?", "-map", "1:a",
         "-c:v", "copy", "-af", filtros,
         "-c:a", "aac", "-b:a", "192k", "-ar", str(SAMPLE_RATE),
         "-shortest", str(salida)])


def salida_por_defecto(video):
    return video.with_name(f"{video.stem}_limpio{video.suffix}")


def decir(msg):
    if sys.stdout:  # en el .exe con ventana no hay consola y sys.stdout es None
        print(msg)


def procesar(video, salida, fuerza, variable, avisar=decir):
    video = Path(video)
    if not video.is_file():
        raise ErrorProceso(f"No existe el archivo: {video}")
    if not 0.0 <= fuerza <= 1.0:
        raise ErrorProceso("La fuerza debe estar entre 0.0 y 1.0")
    if not tiene_audio(video):
        raise ErrorProceso("El video no tiene pista de audio.")

    with tempfile.TemporaryDirectory() as tmp:
        crudo = Path(tmp) / "crudo.wav"
        limpio = Path(tmp) / "limpio.wav"
        avisar("1/3 Extrayendo audio...")
        extraer_audio(video, crudo)
        avisar("2/3 Reduciendo ruido...")
        limpiar(crudo, limpio, fuerza, variable)
        avisar("3/3 Realzando voz y generando video...")
        unir(video, limpio, salida)
    return salida


def ventana():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    raiz = tk.Tk()
    raiz.title("Limpiar voz de video")
    raiz.resizable(False, False)
    marco = ttk.Frame(raiz, padding=16)
    marco.grid()

    ruta = tk.StringVar()
    fuerza = tk.DoubleVar(value=0.85)
    variable = tk.BooleanVar(value=False)
    estado = tk.StringVar(value="Elige un video para empezar.")

    def elegir():
        f = filedialog.askopenfilename(title="Elegir video", filetypes=[
            ("Videos", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.wmv"), ("Todos", "*.*")])
        if f:
            ruta.set(f)

    def iniciar():
        if not ruta.get():
            messagebox.showwarning("Falta video", "Primero elige un video.")
            return
        video = Path(ruta.get())
        salida = filedialog.asksaveasfilename(
            title="Guardar video limpio", initialdir=video.parent,
            initialfile=salida_por_defecto(video).name, defaultextension=video.suffix)
        if not salida:
            return
        boton.state(["disabled"])
        barra.start(12)

        def avisar(msg):
            raiz.after(0, estado.set, msg)

        def trabajo():
            try:
                procesar(video, Path(salida), round(fuerza.get(), 2), variable.get(), avisar)
                raiz.after(0, terminar, None, salida)
            except Exception as e:
                raiz.after(0, terminar, e, salida)

        threading.Thread(target=trabajo, daemon=True).start()

    def terminar(error, salida):
        barra.stop()
        boton.state(["!disabled"])
        if error:
            estado.set("Error.")
            messagebox.showerror("Error", str(error))
        else:
            estado.set(f"Listo: {salida}")
            messagebox.showinfo("Listo", f"Video guardado en:\n{salida}")

    ttk.Label(marco, text="Video:").grid(row=0, column=0, sticky="w")
    ttk.Entry(marco, textvariable=ruta, width=50).grid(row=0, column=1, padx=6)
    ttk.Button(marco, text="Elegir...", command=elegir).grid(row=0, column=2)

    ttk.Label(marco, text="Fuerza:").grid(row=1, column=0, sticky="w", pady=(12, 0))
    texto_fuerza = tk.StringVar(value=f"{fuerza.get():.2f}")
    ttk.Scale(marco, from_=0.3, to=1.0, variable=fuerza,
              command=lambda v: texto_fuerza.set(f"{float(v):.2f}")).grid(
        row=1, column=1, sticky="we", padx=6, pady=(12, 0))
    ttk.Label(marco, textvariable=texto_fuerza, width=5).grid(row=1, column=2, pady=(12, 0))

    ttk.Checkbutton(marco, text="El ruido de fondo cambia mucho (tráfico, gente)",
                    variable=variable).grid(row=2, column=1, sticky="w", pady=8)

    boton = ttk.Button(marco, text="Limpiar audio", command=iniciar)
    boton.grid(row=3, column=1, pady=(4, 8))
    barra = ttk.Progressbar(marco, mode="indeterminate", length=360)
    barra.grid(row=4, column=0, columnspan=3, sticky="we")
    ttk.Label(marco, textvariable=estado, wraplength=440).grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 0))

    raiz.mainloop()


def main():
    if len(sys.argv) == 1:
        ventana()
        return

    p = argparse.ArgumentParser(description="Reduce el ruido de fondo de un video y realza la voz.")
    p.add_argument("video", type=Path, help="video de entrada")
    p.add_argument("-o", "--salida", type=Path, help="video de salida (por defecto: <nombre>_limpio.<ext>)")
    p.add_argument("--fuerza", type=float, default=0.85,
                   help="cuánto ruido quitar, 0.0 a 1.0 (por defecto 0.85; valores altos pueden sonar robóticos)")
    p.add_argument("--ruido-variable", action="store_true",
                   help="usar si el ruido de fondo cambia mucho (tráfico, gente); por defecto se asume ruido constante")
    args = p.parse_args()

    try:
        salida = procesar(args.video, args.salida or salida_por_defecto(args.video),
                          args.fuerza, args.ruido_variable)
    except ErrorProceso as e:
        sys.exit(str(e))
    decir(f"Listo: {salida}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
