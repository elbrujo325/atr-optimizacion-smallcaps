import pandas as pd
import numpy as np
import os
import glob

# --- CONFIGURACIÓN ---
folder_path = r"data"
extension = "*.txt"
ATR_PERIOD = 50
HORIZONTE_VELAS = 20
MUESTRAS_POR_ACTIVO = 500
RIESGO_POR_TRADE = 100
# Rango deseado para BP riesgo 100
BP_MIN = 1200
BP_MAX = 1500

# Ratios TP/SL a analizar
RATIOS_TP_SL = [1.0, 1.5, 2.0]


def calcular_atr(df, period=50):
    """Calcula ATR de forma optimizada"""
    high, low, close = df['High'], df['Low'], df['Close'].shift(1)
    tr = pd.concat([high - low, abs(high - close), abs(low - close)], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calcular_duraciones_multiples_ratios(df, entries_info, coef_sl, ratios):
    """
    Calcula duraciones para múltiples ratios TP/SL en una sola pasada.
    Optimizado para procesar todas las entradas y ratios eficientemente.
    
    Returns: dict con {ratio: [duraciones]}
    """
    resultados = {ratio: [] for ratio in ratios}
    
    for idx, entrada, atr in entries_info:
        sl_level = entrada - coef_sl * atr
        
        # Pre-calcular todos los TP levels
        tp_levels = {ratio: entrada + (ratio * coef_sl * atr) for ratio in ratios}
        
        # Extraer el slice futuro una sola vez
        max_velas = len(df) - idx - 1
        if max_velas <= 0:
            for ratio in ratios:
                resultados[ratio].append(0)
            continue
        
        futuro_high = df['High'].iloc[idx + 1:].values
        futuro_low = df['Low'].iloc[idx + 1:].values
        
        # Para cada ratio, encontrar cuándo toca TP o SL
        for ratio in ratios:
            tp_level = tp_levels[ratio]
            
            # Vectorizado: encontrar primera vela que toca TP o SL
            toca_sl = futuro_low <= sl_level
            toca_tp = futuro_high >= tp_level
            toca_algo = toca_sl | toca_tp
            
            if np.any(toca_algo):
                velas_hasta_hit = np.argmax(toca_algo) + 1
                resultados[ratio].append(velas_hasta_hit)
            else:
                # No tocó nada en el horizonte disponible
                resultados[ratio].append(len(futuro_high))
    
    return resultados


resumen_activos = []
all_files = glob.glob(os.path.join(folder_path, extension))

if not all_files:
    print("No se encontraron archivos.")
else:
    print(f"Iniciando análisis optimizado para {len(all_files)} activos...")
    print(f"Ratios TP/SL a analizar: {RATIOS_TP_SL}\n")
    
    for file_idx, file_path in enumerate(all_files, 1):
        asset_name = os.path.basename(file_path).replace(".txt", "")
        print(f"[{file_idx}/{len(all_files)}] Procesando {asset_name}...", end=" ")
        
        try:
            # Lectura y preparación de datos
            df = pd.read_csv(file_path, sep=",")
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df = df.sort_values('Datetime').reset_index(drop=True)
            df['ATR'] = calcular_atr(df, period=ATR_PERIOD)
            
            indices_validos = list(range(ATR_PERIOD, len(df) - HORIZONTE_VELAS))
            if len(indices_validos) < 10:
                print("❌ Datos insuficientes")
                continue
            
            # =========================
            # FASE 1 — RECOLECCIÓN DE COEFICIENTES SL
            # (Solo calculamos SL, no TP ya que no lo usamos)
            # =========================
            np.random.seed(42)
            candidatos = [i for i in indices_validos if 1 <= df.loc[i, 'Close'] <= 20]
            if len(candidatos) < 10:
                print("❌ Candidatos insuficientes")
                continue
            
            if len(candidatos) < MUESTRAS_POR_ACTIVO:
                muestras_fase1 = candidatos.copy()
            else:
                muestras_fase1 = list(np.random.choice(candidatos, size=MUESTRAS_POR_ACTIVO, replace=False))
            
            # Solo calcular coef_sl (optimización)
            coefs_sl = []
            for idx in muestras_fase1:
                entrada = df.loc[idx, 'Close']
                atr = df.loc[idx, 'ATR']
                if pd.isna(atr) or atr == 0:
                    continue
                
                futuro = df.loc[idx + 1: idx + HORIZONTE_VELAS]
                min_f = futuro['Low'].min()
                c_sl = (entrada - min_f) / atr
                
                if c_sl > 0:
                    coefs_sl.append(c_sl)
            
            if not coefs_sl:
                print("❌ No se generaron coef_sl")
                continue
            
            coefs_sl_sorted = np.sort(np.array(coefs_sl))
            
            # =========================
            # FASE 2 — BUSCAR COEF_SL QUE DA BP EN RANGO
            # =========================
            np.random.seed(7)
            if len(candidatos) < MUESTRAS_POR_ACTIVO:
                muestras_fase2 = candidatos.copy()
            else:
                muestras_fase2 = list(np.random.choice(candidatos, size=MUESTRAS_POR_ACTIVO, replace=False))
            
            coef_sl_seleccionado = None
            bp_promedio_seleccionado = None
            entries_seleccionadas = None
            
            # Búsqueda del coef_sl que da BP en rango
            for candidate_coef_sl in coefs_sl_sorted:
                if candidate_coef_sl <= 0:
                    continue
                
                bp_100 = []
                entries_info = []
                
                for idx in muestras_fase2:
                    entrada = df.loc[idx, 'Close']
                    atr = df.loc[idx, 'ATR']
                    if pd.isna(atr) or atr <= 0:
                        continue
                    
                    distancia_sl = candidate_coef_sl * atr
                    if distancia_sl <= 0:
                        continue
                    
                    shares_100 = 100 / distancia_sl
                    bp_100.append(entrada * shares_100)
                    entries_info.append((idx, entrada, atr))
                
                if not bp_100:
                    continue
                
                bp_mean = np.mean(bp_100)
                
                # Si está en rango, lo seleccionamos
                if BP_MIN <= bp_mean <= BP_MAX:
                    coef_sl_seleccionado = candidate_coef_sl
                    bp_promedio_seleccionado = bp_mean
                    entries_seleccionadas = entries_info.copy()
                    break
            
            # Si no encontró en rango, elegir el más cercano al midpoint
            if coef_sl_seleccionado is None:
                midpoint = (BP_MIN + BP_MAX) / 2
                mejor_dist = None
                mejor_coef = None
                mejor_bp = None
                mejor_entries_info = None
                
                for candidate_coef_sl in coefs_sl_sorted:
                    if candidate_coef_sl <= 0:
                        continue
                    
                    bp_100 = []
                    entries_info = []
                    
                    for idx in muestras_fase2:
                        entrada = df.loc[idx, 'Close']
                        atr = df.loc[idx, 'ATR']
                        if pd.isna(atr) or atr <= 0:
                            continue
                        distancia_sl = candidate_coef_sl * atr
                        if distancia_sl <= 0:
                            continue
                        shares_100 = 100 / distancia_sl
                        bp_100.append(entrada * shares_100)
                        entries_info.append((idx, entrada, atr))
                    
                    if not bp_100:
                        continue
                    
                    bp_mean = np.mean(bp_100)
                    dist = abs(bp_mean - midpoint)
                    
                    if mejor_dist is None or dist < mejor_dist:
                        mejor_dist = dist
                        mejor_coef = candidate_coef_sl
                        mejor_bp = bp_mean
                        mejor_entries_info = entries_info.copy()
                
                coef_sl_seleccionado = mejor_coef
                bp_promedio_seleccionado = mejor_bp
                entries_seleccionadas = mejor_entries_info
            
            if coef_sl_seleccionado is None:
                print("❌ No se encontró coef_sl válido")
                continue
            
            # =========================
            # FASE 3 — CALCULAR DURACIONES PARA MÚLTIPLES RATIOS
            # (Optimizado: calcula todos los ratios en una pasada)
            # =========================
            duraciones_por_ratio = calcular_duraciones_multiples_ratios(
                df, entries_seleccionadas, coef_sl_seleccionado, RATIOS_TP_SL
            )
            
            # Preparar resultado
            resultado = {
                'Activo': asset_name,
                'Coef_SL': coef_sl_seleccionado,
                'BP_Riesgo_100': bp_promedio_seleccionado,
            }
            
            # Añadir columnas para cada ratio
            for ratio in RATIOS_TP_SL:
                duraciones = duraciones_por_ratio[ratio]
                if duraciones:
                    dur_promedio = np.mean(duraciones)
                    resultado[f'TP_{ratio}x'] = ratio * coef_sl_seleccionado
                    resultado[f'Dur_Prom_{ratio}x'] = dur_promedio
                else:
                    resultado[f'TP_{ratio}x'] = ratio * coef_sl_seleccionado
                    resultado[f'Dur_Prom_{ratio}x'] = 0
            
            resumen_activos.append(resultado)
            
            # Mostrar progreso
            dur_str = ', '.join([f"{ratio}x={resultado[f'Dur_Prom_{ratio}x']:.1f}" for ratio in RATIOS_TP_SL])
            print(f"✅ coef_sl={coef_sl_seleccionado:.4f} | BP={bp_promedio_seleccionado:.0f} | Dur: {dur_str}")
        
        except Exception as e:
            print(f"❌ Error: {e}")

# =========================
# RESULTADO FINAL EN TABLA
# =========================
df_final = pd.DataFrame(resumen_activos)

if not df_final.empty:
    # Ordenar por duración promedio del ratio 2x (compatible con versión anterior)
    df_final = df_final.sort_values('Dur_Prom_2.0x')
    
    # Guardar CSV
    output_path = r"result"
    df_final.to_csv(output_path, index=False)
    
    # Mostrar tabla
    pd.options.display.float_format = '{:,.4f}'.format
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    
    print("\n" + "=" * 150)
    print("RESUMEN FINAL (Coeficientes SL + BP + Duraciones para múltiples ratios TP/SL)")
    print("=" * 150)
    print(df_final.to_string(index=False))
    print("=" * 150)
    print(f"\n✅ Resultados guardados en: {output_path}")
    print(f"Total de activos analizados: {len(df_final)}")
else:
    print("\n❌ No se generaron resultados.")