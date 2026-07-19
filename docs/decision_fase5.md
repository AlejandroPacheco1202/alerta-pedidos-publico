# Decisión de Fase 5 — Qué versión va a producción

**Fecha:** julio 2026 · **Período de evaluación:** 42 días hábiles (01/06 – 20/07/2026), split temporal estricto (entrenamiento hasta 31/03/2026).

## Resultados

| Modelo | AUC-ROC | AUC-PR | Precision@15 |
|---|---|---|---|
| Baseline (tasa por día de semana, ventana 12 sem.) | 0.937 | 0.776 | 94.3% |
| Regresión logística (10 features) | 0.937 | 0.780 | 90.0% |
| **Gradient boosting (10 features)** | **0.952** | **0.829** | **96.7%** |

## Decisión: gradient boosting a producción, baseline como respaldo y capa de explicación

### Por qué el boosting

1. **Gana en la métrica de negocio** (+2.4 pp de P@15) y con más claridad en el
   ordenamiento completo de la lista (AUC-PR +0.053), que importa cuando el
   comercial llama más allá del top 15.
2. **La ganancia tiene mecanismo identificable, no es ruido:** el análisis de
   desacuerdos mostró 582 pedidos que el modelo anticipa y el baseline no
   (contra 2 en sentido inverso). Son clientes de frecuencia moderada con
   `ratio_ausencia` ≈ 1: "les toca pedir ya, aunque hoy no sea su día típico".
   Esa interacción día-de-semana × deuda-con-su-ritmo es inexpresable para el
   baseline y para un modelo lineal.
3. **Sus errores son benignos:** los falsos positivos del top-15 (3.3% de los
   lugares) no reinciden, tenían probabilidad ~0.99, y el seguimiento muestra
   que todos siguieron pidiendo con normalidad después. No son fallas del
   modelo: son los pedidos salteados que la herramienta existe para detectar.
   Los falsos negativos (4.1% de los pedidos, score < 0.2) son los clientes
   esporádicos sin patrón que el EDA ya había descartado como objetivo — y
   como piden solos, no cuestan venta.

### Por qué el baseline no se tira

- Queda como **fallback** operativo: si algún día el modelo no puede
  regenerarse (cambio de librerías, corrupción del pickle), el baseline corre
  con pandas puro y sostiene el 94.3%.
- Queda como **capa de explicación**: cada alerta lleva la columna
  `tasa_dia_semana` para que quien llama entienda el motivo en lenguaje de
  negocio ("pidió el 92% de los sábados de los últimos 3 meses").
- Es la **vara de control**: en el monitoreo mensual, si el modelo alguna vez
  rinde por debajo del baseline, es señal de reentrenar o revisar.

### Por qué la regresión logística se descarta

Empata al baseline en AUC pero rinde *peor* en P@15 (90.0%): al ser lineal no
captura la interacción clave y contamina el top de la lista. Documentado como
lección: agregar features sin la clase de modelo adecuada puede empeorar la
métrica que importa.

## Condiciones de operación

- Reentrenamiento: mensual, o ante caída sostenida de precisión vs. baseline.
- La alerta no corre domingos ni feriados (calendario `holidays` AR +
  `data/feriados_extra.csv`).
- Registro de resultado de llamadas (Fase 6) para medir la métrica definitiva:
  **pedidos recuperados por semana**, que ninguna métrica offline puede dar.
