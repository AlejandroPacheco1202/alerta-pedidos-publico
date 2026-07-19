# 🥐 Alerta de pedidos — detección temprana de ventas perdidas

**Sistema de ML que detecta, antes del cierre comercial de las 11:59, qué clientes
habituales todavía no cargaron su pedido del día — para llamarlos a tiempo.**

Proyecto real desplegado en una distribuidora de alimentos (Argentina).
Este repositorio contiene el pipeline completo corriendo sobre un **dataset
sintético** que replica las propiedades estadísticas del real (ver abajo).

## El problema de negocio

Los clientes cargan pedidos cada mañana con cierre 11:59. Con frecuencia, un
cliente habitual "se saltea" su día: olvido, pedido traspapelado entre áreas,
cambio de responsable. Cada caso es venta perdida silenciosa: nadie se entera
hasta que ya es tarde. Un backtest sobre un sábado real encontró **22 puntos de
entrega habituales sin pedido, ninguno de los cuales recuperó ese pedido después**.

## Resultados (sobre los datos reales, 42 días de test con split temporal)

| Modelo | AUC-ROC | AUC-PR | Precision@15 |
|---|---|---|---|
| Baseline (tasa por día de semana) | 0.937 | 0.776 | 94.3% |
| Regresión logística | 0.937 | 0.780 | 90.0% |
| **XGBoost** | **0.951** | **0.829** | **97.1%** |

**Precision@15** = de los 15 clientes que la herramienta prioriza cada día,
cuántos eran efectivamente "pedidores" de ese día. Es la métrica de negocio:
la lista de llamadas tiene costo de tiempo comercial.

## Los tres hallazgos que definieron el diseño

1. **El negocio opera en rutas fijas.** El 80% de los puntos de entrega
   concentra sus pedidos en una de dos rutas (Mar-Jue-Sáb o Lun-Mié-Vie),
   con patrones estables año contra año. La feature `tasa_dia_semana`
   (frecuencia de pedido en ese día de semana, ventana móvil de 12 semanas)
   explica sola la mayor parte de la señal.

2. **Los "falsos positivos" del top-15 son el producto, no un error.** El
   análisis de errores mostró que los clientes alertados que no pidieron
   (3.3% de los lugares del top-15) no reinciden, estaban "en fecha" según su
   ritmo, y todos siguieron comprando normalmente después: son exactamente
   los pedidos salteados que la herramienta existe para recuperar. La métrica
   offline los penaliza; el negocio los celebra.

3. **Un baseline fuerte cambia la conversación.** El baseline de reglas
   (sin ML) alcanza 94.3% de P@15. El valor del XGBoost no es "usar ML":
   son los 582 pedidos del test que anticipa y el baseline no puede ver
   (contra 2 en sentido inverso), concentrados en un perfil identificable:
   clientes en deuda con su propio ritmo fuera de su día más típico.
   La regresión logística, lineal, no captura esa interacción y rinde
   *peor* que el baseline en la métrica de negocio.

## Decisiones de ingeniería

- **Sin data leakage:** todas las features usan `shift(1)`; el perfil que
  predice el día D jamás vio el día D. Evaluación con **split temporal**
  estricto (nunca aleatorio en datos temporales).
- **Feriados:** un feriado no detectado destruía la precisión de un día
  entero (verificado en backtest: 0 pedidos en toda la empresa, 61 alertas
  inútiles). Calendario oficial argentino (`holidays`) + CSV de excepciones.
- **Unidad de análisis: punto de entrega**, no razón social — cada local
  gestiona su propio pedido. Matching de nombres contra el reporte diario
  del ERP con normalización + fuzzy (95% de match exacto).
- **La limpieza se investiga, no se automatiza a ciegas:** las filas sin
  cliente resultaron ser muestras gratis de vendedores a prospectos
  (lista DEMO, importe $0) — excluirlas correctamente requirió entender
  el negocio, no imputar nulos.
- **Baseline en producción como respaldo y capa de explicación:** cada
  alerta lleva su lectura en lenguaje comercial ("pide el 92% de los
  sábados") y el baseline queda como vara de control del drift.

## El producto final

```bash
python src/alerta_diaria.py 2026-07-20 data/raw/Pedidos_20-07-2026.xlsx
```

Genera un Excel formateado para el equipo comercial: lista priorizada por
probabilidad, coloreada por urgencia, con contexto de negocio por cliente
(último pedido, cajas promedio, % histórico de su día). Corre en segundos
sobre el modelo pre-entrenado; el reentrenamiento es mensual.

## Cómo reproducirlo (con datos sintéticos)

```bash
conda env create -f environment.yml
conda activate alerta-pedidos

python src/generar_sintetico.py   # dataset ficticio con las mismas propiedades
python src/load_data.py           # limpieza (encuentra las DEMO y NC plantadas)
python src/features.py            # panel cliente-día anti-leakage
python src/modelo.py              # entrena y compara los 3 enfoques
python src/baseline.py backtest   # backtest del baseline puro
```

Notebooks: `01_eda.ipynb` (análisis exploratorio y hallazgo de rutas),
`02_evaluacion.ipynb` (análisis de errores y decisión de producción).
Documentación de decisiones: `docs/`.

## Sobre el dataset sintético

Los datos reales son confidenciales. `generar_sintetico.py` produce un export
ficticio que replica lo que hace interesante al problema: las dos rutas de
reparto, los cuatro segmentos de frecuencia, la estabilidad de patrones, los
feriados, altas y bajas de clientes, inflación, y hasta el ruido de limpieza
(muestras DEMO sin razón social, notas de crédito). Ningún nombre, dirección
ni importe es real. Los resultados numéricos sobre el sintético son
cualitativamente consistentes con los reales (el boosting supera al baseline
y la logística no).

## Stack

Python · pandas · scikit-learn · XGBoost · openpyxl · matplotlib/seaborn ·
Jupyter · Git. Origen de datos: exports de ERP en Excel (sin acceso a base
de datos — restricción real del entorno, resuelta con un pipeline
reproducible en vez de limpieza manual).

## Impacto

En su primer fin de semana de uso real, la herramienta identificó los pedidos
faltantes del día con nombre y apellido antes del cierre. La métrica
definitiva — pedidos recuperados por semana vía llamadas — está en medición
desde el despliegue.
