import json
import uuid

def cell_id():
    return uuid.uuid4().hex[:8]

def md(source):
    return {
        "cell_type": "markdown",
        "id": cell_id(),
        "metadata": {},
        "source": source if isinstance(source, list) else [source]
    }

def code(source, outputs=None):
    return {
        "cell_type": "code",
        "id": cell_id(),
        "metadata": {},
        "execution_count": None,
        "outputs": outputs or [],
        "source": source if isinstance(source, list) else [source]
    }

cells = []

# ─── PORTADA ────────────────────────────────────────────────────────────────

cells.append(md([
    "# Evaluación Formativa 1 — Análisis Exploratorio e Inferencial\n",
    "\n",
    "**Curso:** MCDI501 — Estadística Computacional para la Toma de Decisiones  \n",
    "**Integrantes:** Carolina Cortés Donoso · Pedro Espinoza Vicentela · Marcelo Corro Troncoso · Juan Pablo Valdebenito Loyola  \n",
    "**Dataset:** Bank Marketing (UCI Machine Learning Repository)  \n",
    "**Fecha:** 24/06/2026  \n",
    "**Repositorio GitHub:** https://github.com/k1ngdom0fbullsh1t/proyecto-grupo4-mcdi501  \n",
]))

# ─── CELDA 0: CONFIGURACIÓN ─────────────────────────────────────────────────

cells.append(md("## Configuración del entorno"))

cells.append(code([
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.ticker as mticker\n",
    "import seaborn as sns\n",
    "from scipy import stats\n",
    "import warnings\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Reproducibilidad\n",
    "SEED = 42\n",
    "np.random.seed(SEED)\n",
    "\n",
    "# Estilo global de gráficos\n",
    "plt.rcParams.update({\n",
    "    'figure.figsize': (10, 5),\n",
    "    'axes.titlesize': 13,\n",
    "    'axes.labelsize': 11,\n",
    "    'xtick.labelsize': 10,\n",
    "    'ytick.labelsize': 10,\n",
    "    'figure.dpi': 120,\n",
    "})\n",
    "sns.set_theme(style='whitegrid', palette='muted')\n",
    "\n",
    "print('Entorno configurado correctamente.')",
]))

# ─── SECCIÓN 1 ───────────────────────────────────────────────────────────────

cells.append(md([
    "---\n",
    "## Sección 1 — Preparación y carga de datos\n",
    "\n",
    "> Carga del dataset, verificación de estructura, tipos de variables, reporte de calidad y limpieza básica.\n",
]))

cells.append(md("### 1.1 Carga del dataset"))

cells.append(code([
    "from pathlib import Path\n",
    "\n",
    "# Detectar raíz del proyecto buscando la carpeta data/ hacia arriba\n",
    "raiz = Path().resolve()\n",
    "for _ in range(5):\n",
    "    if (raiz / 'data').exists():\n",
    "        break\n",
    "    raiz = raiz.parent\n",
    "\n",
    "RUTA = raiz / 'data' / 'raw' / 'bank-additional' / 'bank-additional-full.csv'\n",
    "df = pd.read_csv(RUTA, sep=';')\n",
    "\n",
    "print(f'Shape: {df.shape}')\n",
    "print(f'Filas: {df.shape[0]:,} | Columnas: {df.shape[1]}')\n",
    "df.head(3)",
]))

cells.append(md("### 1.2 Tipos de variables"))

cells.append(code([
    "tipos_tabla = {\n",
    "    'Variable':    ['age','job','marital','education','default','housing','loan',\n",
    "                    'contact','month','day_of_week','duration','campaign',\n",
    "                    'pdays','previous','poutcome',\n",
    "                    'emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed',\n",
    "                    'y'],\n",
    "    'Tipo estadístico': [\n",
    "        'Numérica discreta','Categórica nominal','Categórica nominal','Categórica ordinal',\n",
    "        'Categórica nominal','Categórica nominal','Categórica nominal',\n",
    "        'Categórica nominal','Categórica nominal','Categórica nominal',\n",
    "        'Numérica continua','Numérica discreta',\n",
    "        'Numérica discreta','Numérica discreta','Categórica nominal',\n",
    "        'Numérica continua','Numérica continua','Numérica continua','Numérica continua','Numérica continua',\n",
    "        'Binaria (objetivo)'\n",
    "    ],\n",
    "    'Tipo Python': [str(df[c].dtype) for c in\n",
    "        ['age','job','marital','education','default','housing','loan',\n",
    "         'contact','month','day_of_week','duration','campaign',\n",
    "         'pdays','previous','poutcome',\n",
    "         'emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','y']],\n",
    "    'Descripción': [\n",
    "        'Edad del cliente','Ocupación laboral','Estado civil','Nivel educativo',\n",
    "        'Tiene crédito en mora','Tiene crédito hipotecario','Tiene préstamo personal',\n",
    "        'Canal de contacto (celular/teléfono)','Mes de último contacto','Día de la semana',\n",
    "        'Duración última llamada (seg.) — proxy del target, excluir de modelos','N° contactos en esta campaña',\n",
    "        'Días desde contacto previo (999 = nunca contactado)','N° contactos antes de esta campaña',\n",
    "        'Resultado campaña anterior',\n",
    "        'Tasa variación empleo (trim.)','Índice precios al consumidor (mens.)','Índice confianza consumidor (mens.)',\n",
    "        'Tasa euribor 3 meses (diaria)','N° empleados (trim.)',\n",
    "        '¿Suscribió depósito a plazo? (no/yes)'\n",
    "    ]\n",
    "}\n",
    "pd.DataFrame(tipos_tabla).set_index('Variable')",
]))

cells.append(md("### 1.3 Reporte de calidad"))

cells.append(code([
    "# Valores NaN reales\n",
    "nan_real = df.isnull().sum()\n",
    "print('=== Valores NaN reales ===')\n",
    "print(nan_real[nan_real > 0] if nan_real.sum() > 0 else 'Sin valores NaN en ninguna columna.')\n",
    "\n",
    "# Valores 'unknown' por columna (solo categóricas)\n",
    "print('\\n=== Valores \"unknown\" por columna ===')\n",
    "categoricas = df.select_dtypes(include='object').columns\n",
    "unknown_df = pd.DataFrame({\n",
    "    'Columna': categoricas,\n",
    "    'N unknown': [df[c].eq('unknown').sum() for c in categoricas],\n",
    "    '% unknown': [(df[c].eq('unknown').sum() / len(df) * 100).round(2) for c in categoricas]\n",
    "}).set_index('Columna')\n",
    "unknown_df = unknown_df[unknown_df['N unknown'] > 0]\n",
    "print(unknown_df.to_string() if len(unknown_df) > 0 else 'Sin valores unknown.')\n",
    "\n",
    "# Duplicados\n",
    "dups = df.duplicated().sum()\n",
    "print(f'\\n=== Duplicados === {dups} filas duplicadas')\n",
    "\n",
    "# Inconsistencias conocidas\n",
    "dur_cero = (df['duration'] == 0).sum()\n",
    "pdays_999 = (df['pdays'] == 999).sum()\n",
    "print(f'\\n=== Inconsistencias conocidas ===')\n",
    "print(f'duration == 0: {dur_cero} registros (llamadas sin contacto real)')\n",
    "print(f'pdays == 999: {pdays_999} registros ({pdays_999/len(df)*100:.1f}%) — nunca contactados antes')",
]))

cells.append(md("### 1.4 Limpieza básica"))

cells.append(code([
    "df_clean = df.copy()\n",
    "\n",
    "# Codificar variable objetivo como binaria\n",
    "df_clean['y_bin'] = df_clean['y'].map({'no': 0, 'yes': 1})\n",
    "\n",
    "# Tabla resumen de calidad antes/después\n",
    "resumen = pd.DataFrame({\n",
    "    'Métrica': ['Filas totales', 'Columnas', 'Valores NaN', 'Duplicados',\n",
    "                'duration == 0', 'pdays == 999', 'y codificada (0/1)'],\n",
    "    'Antes': [len(df), df.shape[1], df.isnull().sum().sum(), df.duplicated().sum(),\n",
    "              (df['duration'] == 0).sum(), (df['pdays'] == 999).sum(), 'No'],\n",
    "    'Después': [len(df_clean), df_clean.shape[1], df_clean.isnull().sum().sum(),\n",
    "                df_clean.duplicated(subset=df.columns).sum(),\n",
    "                (df_clean['duration'] == 0).sum(), (df_clean['pdays'] == 999).sum(), 'Sí (columna y_bin)'],\n",
    "    'Decisión': [\n",
    "        '—', '—', 'No se imputan (no hay NaN reales)', 'Se mantienen (sin duplicados exactos)',\n",
    "        'Se mantienen para EDA; excluir en modelos',\n",
    "        'Se mantienen; representan clientes sin historial previo',\n",
    "        'Agregada columna y_bin para análisis numérico'\n",
    "    ]\n",
    "})\n",
    "resumen.set_index('Métrica')",
]))

# ─── SECCIÓN 2 ───────────────────────────────────────────────────────────────

cells.append(md([
    "---\n",
    "## Sección 2 — Análisis exploratorio de datos (EDA)\n",
    "\n",
    "> Estadística descriptiva de variables numéricas y categóricas, visualizaciones y análisis bivariado.\n",
]))

cells.append(md("### 2.1 Variables numéricas — Estadística descriptiva"))

cells.append(code([
    "num_cols = ['age','duration','campaign','pdays','previous',\n",
    "            'emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']\n",
    "\n",
    "desc = df_clean[num_cols].describe().T\n",
    "desc['cv'] = (desc['std'] / desc['mean']).round(3)\n",
    "desc.round(3)",
]))

cells.append(md("### 2.2 Distribuciones — Histogramas y boxplots"))

cells.append(code([
    "fig, axes = plt.subplots(2, 5, figsize=(20, 8))\n",
    "axes = axes.flatten()\n",
    "\n",
    "for i, col in enumerate(num_cols):\n",
    "    axes[i].hist(df_clean[col], bins=40, color='steelblue', edgecolor='white', alpha=0.85)\n",
    "    axes[i].set_title(col)\n",
    "\n",
    "fig.suptitle('Distribución de variables numéricas', fontsize=14, y=1.01)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "fig2, axes2 = plt.subplots(2, 5, figsize=(20, 8))\n",
    "axes2 = axes2.flatten()\n",
    "\n",
    "for i, col in enumerate(num_cols):\n",
    "    axes2[i].boxplot(df_clean[col], vert=True, patch_artist=True,\n",
    "                     boxprops=dict(facecolor='steelblue', alpha=0.6))\n",
    "    axes2[i].set_title(col)\n",
    "\n",
    "fig2.suptitle('Boxplots de variables numéricas', fontsize=14, y=1.01)\n",
    "plt.tight_layout()\n",
    "plt.show()",
]))

cells.append(md("### 2.3 Variables categóricas — Frecuencias"))

cells.append(code([
    "cat_cols = ['job','marital','education','default','housing','loan',\n",
    "            'contact','month','day_of_week','poutcome','y']\n",
    "\n",
    "for col in cat_cols:\n",
    "    freq = df_clean[col].value_counts()\n",
    "    freq_rel = (freq / len(df_clean) * 100).round(2)\n",
    "    tabla = pd.DataFrame({'N': freq, '%': freq_rel})\n",
    "    print(f'\\n--- {col} ---')\n",
    "    print(tabla.to_string())",
]))

cells.append(md("### 2.4 Barplots de variables categóricas"))

cells.append(code([
    "fig, axes = plt.subplots(3, 4, figsize=(22, 14))\n",
    "axes = axes.flatten()\n",
    "\n",
    "for i, col in enumerate(cat_cols):\n",
    "    orden = df_clean[col].value_counts().index\n",
    "    ax = axes[i]\n",
    "    counts = df_clean[col].value_counts()[orden]\n",
    "    ax.bar(range(len(orden)), counts.values, color='steelblue', alpha=0.85, edgecolor='white')\n",
    "    ax.set_xticks(range(len(orden)))\n",
    "    ax.set_xticklabels(orden, rotation=45, ha='right', fontsize=9)\n",
    "    ax.set_title(col)\n",
    "    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))\n",
    "\n",
    "axes[-1].set_visible(False)\n",
    "fig.suptitle('Frecuencia de variables categóricas', fontsize=14, y=1.01)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(f'\\nDesbalance de clase objetivo (y):')\n",
    "print(df_clean['y'].value_counts(normalize=True).mul(100).round(2).to_string())",
]))

cells.append(md("### 2.5 Análisis bivariado — Tasa de suscripción por variable categórica"))

cells.append(code([
    "vars_bivariado = ['job', 'education', 'month', 'poutcome']\n",
    "fig, axes = plt.subplots(2, 2, figsize=(18, 12))\n",
    "axes = axes.flatten()\n",
    "\n",
    "for i, col in enumerate(vars_bivariado):\n",
    "    tasa = df_clean.groupby(col)['y_bin'].mean().sort_values(ascending=False) * 100\n",
    "    ax = axes[i]\n",
    "    ax.bar(range(len(tasa)), tasa.values, color='steelblue', alpha=0.85, edgecolor='white')\n",
    "    ax.set_xticks(range(len(tasa)))\n",
    "    ax.set_xticklabels(tasa.index, rotation=45, ha='right', fontsize=10)\n",
    "    ax.set_title(f'Tasa de suscripción por {col} (%)')\n",
    "    ax.set_ylabel('% suscripción')\n",
    "    ax.axhline(y=df_clean['y_bin'].mean()*100, color='tomato', linestyle='--', label='Media global')\n",
    "    ax.legend(fontsize=9)\n",
    "    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1f}%'))\n",
    "\n",
    "fig.suptitle('Tasa de suscripción a depósito a plazo por variable categórica', fontsize=13)\n",
    "plt.tight_layout()\n",
    "plt.show()",
]))

cells.append(md([
    "**Interpretación:** El resultado de la campaña anterior (`poutcome`) es la variable categórica con mayor poder discriminante: ",
    "los clientes con resultado `success` presentan una tasa de suscripción notablemente superior a la media global (~11.3%). ",
    "En cuanto al mes de contacto, `mar`, `dec` y `sep` muestran tasas elevadas, posiblemente asociadas a condiciones ",
    "macroeconómicas favorables en esos períodos. Por ocupación, los estudiantes y jubilados presentan mayor propensión ",
    "a suscribir que ocupaciones como `blue-collar` o `services`.\n"
]))

cells.append(md("### 2.6 Correlación entre variables numéricas"))

cells.append(code([
    "corr_cols = ['age','duration','campaign','previous','emp.var.rate',\n",
    "             'cons.price.idx','cons.conf.idx','euribor3m','nr.employed','y_bin']\n",
    "\n",
    "corr = df_clean[corr_cols].corr()\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(11, 9))\n",
    "mask = np.triu(np.ones_like(corr, dtype=bool))\n",
    "sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',\n",
    "            center=0, square=True, linewidths=0.5, ax=ax,\n",
    "            cbar_kws={'shrink': 0.8})\n",
    "ax.set_title('Matriz de correlación — variables numéricas', fontsize=13)\n",
    "plt.tight_layout()\n",
    "plt.show()",
]))

cells.append(md([
    "**Interpretación:** Se observa alta correlación positiva entre `emp.var.rate`, `euribor3m` y `nr.employed` ",
    "(r > 0.9), lo que indica que estas variables macroeconómicas se mueven conjuntamente y podrían generar ",
    "multicolinealidad en modelos futuros. La variable `y_bin` presenta correlaciones moderadas-bajas con todas ",
    "las numéricas, siendo `euribor3m` (negativa) y `previous` (positiva) las de mayor magnitud relativa. ",
    "`duration` muestra correlación positiva con `y_bin`, pero debe excluirse de modelos predictivos por ser ",
    "un proxy post-hoc del resultado.\n"
]))

cells.append(md("### 2.7 Distribución de variables numéricas clave por clase objetivo"))

cells.append(code([
    "vars_box = ['age', 'euribor3m', 'nr.employed']\n",
    "fig, axes = plt.subplots(1, 3, figsize=(16, 6))\n",
    "\n",
    "for i, col in enumerate(vars_box):\n",
    "    data_no  = df_clean[df_clean['y'] == 'no'][col]\n",
    "    data_yes = df_clean[df_clean['y'] == 'yes'][col]\n",
    "    ax = axes[i]\n",
    "    ax.boxplot([data_no, data_yes], tick_labels=['no', 'yes'], patch_artist=True,\n",
    "               boxprops=dict(alpha=0.7),\n",
    "               medianprops=dict(color='black', linewidth=2))\n",
    "    ax.set_title(f'{col} por clase (y)')\n",
    "    ax.set_xlabel('Suscribió depósito')\n",
    "    ax.set_ylabel(col)\n",
    "\n",
    "fig.suptitle('Distribución de variables numéricas clave según suscripción', fontsize=13)\n",
    "plt.tight_layout()\n",
    "plt.show()",
]))

cells.append(md([
    "**Interpretación:** Los clientes que suscriben (`yes`) tienden a ser contactados en contextos de menor tasa ",
    "euribor y menor número de empleados, sugiriendo que las campañas son más efectivas en períodos de menor ",
    "actividad económica o tasas de interés bajas. La distribución de edad es similar entre ambas clases, ",
    "aunque con ligera diferencia en la mediana.\n"
]))

# ─── NOTEBOOK FINAL ─────────────────────────────────────────────────────────

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13.0"
        }
    },
    "cells": cells
}

ruta = r"notebooks\f1_definicion\F1_Definicion.ipynb"
with open(ruta, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Notebook generado: {ruta}")
