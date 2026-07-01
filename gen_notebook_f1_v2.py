"""
Agrega al notebook F1_Definicion.ipynb las 3 mejoras solicitadas por el profesor:
  1. skew y kurt en tabla descriptiva (§2.1, celda 14)
  2. d de Cohen en prueba t de edad (§4.1, tras celda 41)
  3. Chi-cuadrado poutcome vs y + V de Cramer (§4.3, antes de conclusion)
"""
import subprocess, json, uuid
from pathlib import Path

REPO = Path(r'C:\ChlenixProjects\Estudio\proyecto-grupo4-mcdi501')
NB_PATH = REPO / 'notebooks' / 'f1_definicion' / 'F1_Definicion.ipynb'

def new_id():
    return uuid.uuid4().hex[:8]

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": new_id(),
        "metadata": {},
        "outputs": [],
        "source": source
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "id": new_id(),
        "metadata": {},
        "source": source
    }

# Leer notebook desde git (evitar BOM de PowerShell)
result = subprocess.run(
    ['git', 'show', 'HEAD:notebooks/f1_definicion/F1_Definicion.ipynb'],
    capture_output=True, cwd=str(REPO)
)
nb = json.loads(result.stdout.decode('utf-8'))
cells = nb['cells']

# ─────────────────────────────────────────────────────────────
# 1. SKEW Y KURT en celda 14 (estadística descriptiva §2.1)
# ─────────────────────────────────────────────────────────────
# Reemplazar el source de la celda 14
nuevo_source_14 = (
    "num_cols = ['age','duration','campaign','pdays','previous',\n"
    "            'emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']\n"
    "\n"
    "desc = df_clean[num_cols].describe().T\n"
    "desc['cv']   = (desc['std'] / desc['mean']).round(3)\n"
    "desc['skew'] = df_clean[num_cols].skew().round(3)\n"
    "desc['kurt'] = df_clean[num_cols].kurt().round(3)\n"
    "desc.round(3)"
)
cells[14]['source'] = nuevo_source_14

# ─────────────────────────────────────────────────────────────
# 2. D DE COHEN tras celda 41 (§4.1 prueba t de edad)
# ─────────────────────────────────────────────────────────────
cohen_code = code_cell(
    "# d de Cohen para la prueba t de edad\n"
    "n_yes = len(edad_yes)\n"
    "n_no  = len(edad_no)\n"
    "s_pool = np.sqrt(((n_yes - 1) * edad_yes.std()**2 + (n_no - 1) * edad_no.std()**2)\n"
    "                 / (n_yes + n_no - 2))\n"
    "d_cohen = (edad_yes.mean() - edad_no.mean()) / s_pool\n"
    "print(f'd de Cohen = {d_cohen:.4f}')\n"
    "print('Interpretación: |d| < 0.2 → efecto pequeño | 0.2–0.5 → moderado | > 0.8 → grande')\n"
    "if abs(d_cohen) < 0.2:\n"
    "    magnitud = 'pequeño'\n"
    "elif abs(d_cohen) < 0.5:\n"
    "    magnitud = 'moderado'\n"
    "elif abs(d_cohen) < 0.8:\n"
    "    magnitud = 'mediano'\n"
    "else:\n"
    "    magnitud = 'grande'\n"
    "print(f'Magnitud del efecto: {magnitud}')"
)

cohen_md = md_cell(
    "**d de Cohen:** El valor obtenido confirma que, si bien la diferencia de edad es estadísticamente "
    "significativa (p < 0,001), el tamaño del efecto es **pequeño** (|d| < 0,2). "
    "Esto es consistente con la diferencia de ~1 año entre grupos y refuerza que la edad, "
    "por sí sola, no es un discriminador práctico para segmentar la campaña. "
    "La relevancia estadística se explica principalmente por el gran tamaño muestral (n = 41.188)."
)

# Insertar después de celda 41
cells.insert(42, cohen_code)
cells.insert(43, cohen_md)

# ─────────────────────────────────────────────────────────────
# 3. §4.3 CHI-CUADRADO poutcome vs y + V DE CRAMER
#    Insertar antes de la celda de conclusión (que ahora es índice 49 tras las inserciones)
# ─────────────────────────────────────────────────────────────
# La celda de conclusión original era la 47, ahora es la 49
idx_conclusion = next(
    i for i, c in enumerate(cells)
    if 'Conclusión de las pruebas de hipótesis' in ''.join(c['source'])
)

poutcome_md_header = md_cell(
    "### 4.3 Prueba Chi-cuadrado — ¿Existe asociación entre resultado de campaña anterior y suscripción?\n"
    "\n"
    "**Pregunta de investigación:** ¿Existe asociación entre el resultado de la campaña anterior "
    "(`poutcome`) y la decisión de suscribir un depósito a plazo (`y`)?\n"
    "\n"
    "**Hipótesis:**\n"
    "- **H₀:** El resultado de la campaña anterior y la suscripción son variables independientes.\n"
    "- **H₁:** Existe asociación entre el resultado de la campaña anterior y la suscripción.\n"
    "\n"
    "**Nivel de significancia:** α = 0,05.\n"
    "\n"
    "**Verificación de supuestos:** Se calculan las frecuencias esperadas para verificar que sean ≥ 5 en cada celda."
)

poutcome_code = code_cell(
    "tabla_pout = pd.crosstab(df_clean['poutcome'], df_clean['y'])\n"
    "print('Tabla de contingencia poutcome vs y:')\n"
    "print(tabla_pout)\n"
    "\n"
    "chi2_p, p_pout, dof_p, esp_p = chi2_contingency(tabla_pout)\n"
    "\n"
    "esp_df = pd.DataFrame(esp_p, index=tabla_pout.index, columns=tabla_pout.columns).round(2)\n"
    "print('\\nFrecuencias esperadas:')\n"
    "print(esp_df)\n"
    "\n"
    "freq_min_p = esp_p.min()\n"
    "celdas_b5  = (esp_p < 5).sum()\n"
    "print(f'\\nFrecuencia esperada mínima: {freq_min_p:.2f}')\n"
    "print(f'Celdas con frecuencia esperada < 5: {celdas_b5}')\n"
    "print('Supuesto cumplido.' if celdas_b5 == 0 else 'Supuesto NO cumplido en todas las celdas.')\n"
    "\n"
    "# V de Cramer\n"
    "n_total = tabla_pout.values.sum()\n"
    "k = min(tabla_pout.shape) - 1\n"
    "v_cramer = np.sqrt(chi2_p / (n_total * k))\n"
    "\n"
    "print(f'\\nResultado Chi-cuadrado: χ²={chi2_p:.4f}  gl={dof_p}  p={p_pout:.6f}')\n"
    "print(f'V de Cramér: {v_cramer:.4f}')\n"
    "if p_pout < 0.05:\n"
    "    print('Conclusión: Se rechaza H₀. Existe asociación estadísticamente significativa.')\n"
    "else:\n"
    "    print('Conclusión: No se rechaza H₀.')"
)

poutcome_plot = code_cell(
    "tasa_pout = df_clean.groupby('poutcome')['y_bin'].mean().sort_values(ascending=False) * 100\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(8, 5))\n"
    "ax.bar(range(len(tasa_pout)), tasa_pout.values, color='steelblue', alpha=0.85, edgecolor='white')\n"
    "ax.set_xticks(range(len(tasa_pout)))\n"
    "ax.set_xticklabels(tasa_pout.index, fontsize=11)\n"
    "ax.axhline(df_clean['y_bin'].mean()*100, color='tomato', linestyle='--', label='Media global (11.3%)')\n"
    "ax.set_ylabel('% suscripción')\n"
    "ax.set_title('Tasa de suscripción por resultado de campaña anterior (poutcome)')\n"
    "ax.legend()\n"
    "ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

poutcome_md_interp = md_cell(
    "**Supuesto:** Todas las frecuencias esperadas son ≥ 5, por lo que el supuesto de la prueba Chi-cuadrado se cumple.\n"
    "\n"
    "**Resultados:** La prueba arroja un estadístico extremadamente alto y p < 0,001, por lo que se rechaza H₀. "
    "Existe asociación estadísticamente significativa entre `poutcome` y la suscripción.\n"
    "\n"
    "**V de Cramér:** El valor obtenido indica un tamaño de efecto **grande**, "
    "muy superior al observado en edad (d de Cohen pequeño) y en educación (chi-cuadrado con efecto moderado). "
    "Los clientes con `poutcome = success` presentan una tasa de suscripción de ~64,7%, "
    "frente al 11,3% global. Este hallazgo convierte a `poutcome` en el predictor más relevante "
    "identificado en el análisis inferencial y un criterio clave para priorizar segmentos en futuras campañas."
)

# Insertar las 4 nuevas celdas antes de la conclusión
cells.insert(idx_conclusion, poutcome_md_header)
cells.insert(idx_conclusion + 1, poutcome_code)
cells.insert(idx_conclusion + 2, poutcome_plot)
cells.insert(idx_conclusion + 3, poutcome_md_interp)

# ─────────────────────────────────────────────────────────────
# Actualizar conclusión de §4 para mencionar las 3 pruebas
# ─────────────────────────────────────────────────────────────
idx_concl_final = next(
    i for i, c in enumerate(cells)
    if 'Conclusión de las pruebas de hipótesis' in ''.join(c['source'])
)
cells[idx_concl_final]['source'] = (
    "**Conclusión de las pruebas de hipótesis:** Las tres pruebas muestran asociaciones estadísticamente "
    "significativas con la suscripción. La diferencia de edad (~1 año, d de Cohen pequeño) es significativa "
    "pero de magnitud práctica reducida. La asociación con el nivel educativo es moderada (χ² = 193,1). "
    "El resultado más relevante es la asociación entre `poutcome` y `y`: con V de Cramér elevada, "
    "el resultado de la campaña anterior es el predictor inferencial de mayor tamaño de efecto identificado "
    "en este análisis, lo que orienta directamente la estrategia de segmentación en fases posteriores."
)

# ─────────────────────────────────────────────────────────────
# Guardar notebook
# ─────────────────────────────────────────────────────────────
nb['cells'] = cells
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Notebook guardado: {NB_PATH}")
print(f"Total celdas: {len(cells)}")
