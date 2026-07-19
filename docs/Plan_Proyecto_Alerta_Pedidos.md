# Plan de Proyecto: Alerta de Clientes sin Pedido antes del Cierre

**Objetivo de negocio:** cada día hábil, antes de las 11:59, generar una lista priorizada de clientes que, según su patrón histórico, deberían haber cargado un pedido y todavía no lo hicieron, para que el equipo comercial los contacte.

**Objetivo de portfolio:** demostrar el ciclo completo de un proyecto de datos real: limpieza de exports de ERP, análisis exploratorio, baseline estadístico (RFM/reglas), modelo de machine learning, evaluación comparativa honesta y puesta en producción, más una versión publicable sin datos confidenciales.

**Stack:** Python (pandas, matplotlib/seaborn, scikit-learn, XGBoost), Jupyter para exploración, scripts `.py` para el pipeline estable, Git para versionado.

---

## Radiografía inicial de los datos (ya verificada)

| Aspecto | Valor |
|---|---|
| Fuente histórica | `Informe_de_ventas_diario_2025.xlsx`, hoja `base venta` |
| Filas | 233.437 |
| Período | 02/01/2025 → 20/07/2026 (18,5 meses) |
| Clientes únicos (razón social) | 1.235 |
| Puntos de entrega únicos | 1.694 |
| Columnas | Fecha, Tipo de Cliente, Lista Precios, Vendedor, Cliente - Razón Social, Nombre Fantasía, Dirección de Entrega, Localidad, Rubro, Artículo, Tipo de comprobante, Cajas, Importe Neto, Importe Descuento, Bonificación, Mes |
| Nulos relevantes | ~2.400 filas (~1%) sin razón social / fantasía / dirección |
| Comprobantes | 92% Factura; también Facturas Mi Pyme, Facturas 0, Exportación, y Notas de crédito/débito |
| Reporte diario | `Pedidos_DD-MM-AAAA.xlsx`: fecha (serial Excel), Cliente, Lugar de Entrega, Comprobante (NP), Artículo, Cantidad |

**Granularidad:** una fila = un artículo dentro de un comprobante. Un "pedido" es la agregación de filas por cliente + fecha (+ punto de entrega).

---

## Decisiones de diseño documentadas

### Decisión 1 — Unidad de análisis: ¿cliente o punto de entrega?

**Escenarios:**
- (a) Razón social: 1.235 entidades. Simple, pero una cadena con 10 locales que pide desde 9 quedaría "cubierta" aunque un local se haya olvidado.
- (b) Punto de entrega (Nombre Fantasía / Dirección): 1.694 entidades. Refleja cómo se pide en la realidad (cada local carga su nota de pedido, como se ve en el reporte diario donde 819 CAFE SRL tiene un NP distinto por local).
- (c) Híbrido: modelar por punto de entrega y reportar agrupado por razón social.

**Elegimos (c) con foco en (b):** el evento real es "el local X no pidió", y el reporte para el comercial se agrupa por razón social para facilitar la llamada. Costo: algo más de trabajo de agregación. Beneficio: la alerta apunta exactamente a lo que se perdió.

### Decisión 2 — Qué cuenta como "pedido" en el histórico

**Escenarios:**
- (a) Todas las filas.
- (b) Solo comprobantes de venta (Facturas en todas sus variantes), excluyendo notas de crédito y débito.

**Elegimos (b):** una nota de crédito es una devolución o ajuste, no un pedido; incluirla inventaría "días de compra" falsos. Verificación: comparar cantidad de días-cliente antes y después del filtro y revisar 5 casos a mano.

### Decisión 3 — Definición del target (la variable a predecir)

**Escenarios:**
- (a) "¿Cuántas cajas va a pedir hoy?" (regresión).
- (b) "¿Va a pedir hoy sí o no?" (clasificación binaria por cliente-día).

**Elegimos (b):** el negocio necesita una lista de llamadas, no un pronóstico de volumen. La clasificación binaria es más simple, más robusta y su salida (probabilidad) ordena la lista directamente. La regresión de volumen puede ser una extensión futura.

**Construcción del dataset de entrenamiento:** para cada punto de entrega activo y cada día hábil del período, una fila con target = 1 si facturó ese día, 0 si no. Esto convierte 233 mil filas de ventas en un panel de ~cientos de miles de observaciones cliente-día.

### Decisión 4 — Clientes activos vs. dados de baja

**Problema:** un cliente que dejó de comprar en marzo 2025 no debe generar alertas en 2026, pero tampoco queremos perder clientes estacionales.

**Escenarios:**
- (a) Ventana fija: activo = compró en los últimos N días (ej. 60).
- (b) Ventana relativa a su frecuencia: activo = su silencio actual no supera K veces su intervalo típico entre pedidos.

**Elegimos (b) con (a) como tope:** un cliente diario que lleva 10 días sin pedir está "dormido"; uno quincenal con 10 días de silencio está normal. La regla relativa respeta el ritmo de cada uno. Verificación: listar los clientes excluidos y validarlos contra el conocimiento del negocio (vos).

### Decisión 5 — Cómo evitar data leakage (concepto clave, explicado abajo)

**Regla:** todas las features de un cliente-día se calculan solo con información disponible **antes** de ese día. Nunca usar datos del mismo día ni del futuro. En código: las ventanas móviles se calculan con `shift(1)` antes de agregar.

### Decisión 6 — Cómo evaluar (split temporal, no aleatorio)

**Escenarios:**
- (a) Split aleatorio 80/20 clásico de los cursos.
- (b) Split temporal: entrenar con ene-2025 → mar-2026, validar abr-may 2026, testear jun-jul 2026.

**Elegimos (b):** con datos temporales, el split aleatorio filtra información del futuro al entrenamiento (leakage) y da métricas infladas que no se sostienen en producción. El split temporal simula exactamente el uso real: predecir mañana con lo que sabías hasta hoy. Este es uno de los puntos que más criterio demuestra en un portfolio.

**Métricas:**
- **Precision@15**: de los 15 clientes que la herramienta manda a llamar, ¿cuántos efectivamente eran "pedidores" de ese día? Es LA métrica de negocio (la lista de llamadas tiene costo de tiempo comercial).
- AUC-ROC y AUC-PR como métricas técnicas generales.
- Comparación siempre contra el baseline RFM: el modelo ML solo se justifica si le gana.

### Decisión 7 — Matching de nombres entre reporte diario e histórico

**Problema detectado:** el reporte diario usa "Cliente" + "Lugar de Entrega" (ej. "819 CAFE SRL" / "CM ABASTO") mientras el histórico separa "Razón Social" / "Nombre Fantasía" / "Dirección". Los textos pueden no coincidir exactamente.

**Plan:** normalizar (mayúsculas, tildes, espacios, puntuación, sufijos societarios S.A./SRL) y matchear primero exacto, luego fuzzy (rapidfuzz) solo para los no matcheados, con revisión manual de ese remanente. Verificación: % de matcheo exacto reportado; objetivo ≥95% antes de avanzar.

---

## Preguntas abiertas a validar con el negocio (vos)

1. **¿La columna Fecha del histórico es fecha de pedido o de facturación/entrega?** Si hay corrimiento (pido hoy, facturo mañana), el patrón semanal se desplaza un día. No bloquea el desarrollo pero hay que saberlo para interpretar.
2. **¿Existe un export parcial intra-mañana?** El reporte de pedidos llega después de las 11:59; para llamar a las 10:00 se necesita un corte parcial (aunque sea manual). Alternativa provisoria: correr la alerta a primera hora con "quiénes suelen pedir hoy" sin descontar los ya cargados.
3. **¿Feriados y días no laborables?** Confirmar si se factura sábados (el reporte del 18/07/2026 es sábado, aparentemente sí) y armar el calendario de feriados argentinos para las features.

---

## Fases del proyecto

### Fase 0 — Setup del proyecto (½ día)
- Crear repo Git con estructura: `data/raw/` (gitignored), `data/processed/`, `notebooks/`, `src/`, `outputs/`, `README.md`.
- Entorno: `requirements.txt` con pandas, openpyxl, matplotlib, seaborn, scikit-learn, xgboost, rapidfuzz.
- **Verificable:** el repo existe, `pip install -r requirements.txt` corre limpio, y el README ya cuenta el problema de negocio en 3 párrafos.

### Fase 1 — Ingesta y limpieza (1-2 días)
- Script `src/load_data.py`: lee `base venta`, tipa columnas, parsea fechas (incluyendo seriales de Excel), normaliza textos de cliente.
- Tratar los ~2.400 nulos de razón social: investigar qué son (¿anulaciones?, ¿un tipo de comprobante?) y decidir excluir o imputar, documentando.
- Filtrar comprobantes según Decisión 2.
- Guardar como `data/processed/ventas.parquet` (formato columnar: carga en segundos vs. el minuto del Excel).
- **Verificable:** un notebook de validación que imprime: filas antes/después de cada filtro, rango de fechas, clientes únicos, y 0 fechas inválidas. Cada número tiene que poder explicarse.

### Fase 2 — Análisis exploratorio (EDA) (2-3 días)
- Distribución de pedidos por día de semana, global y por cliente.
- Distribución de intervalos entre pedidos por cliente (el insumo de la Decisión 4).
- Segmentación básica: ¿cuántos clientes son diarios, semanales, quincenales, erráticos?
- Estacionalidad mensual y efecto feriados.
- Identificar el universo de clientes activos.
- **Verificable:** notebook con 6-8 gráficos comentados y una tabla resumen de segmentos de frecuencia. Este notebook es pieza central del portfolio.

### Fase 3 — Baseline RFM / reglas (2-3 días)
- Para cada punto de entrega: perfil de días de semana en que pide (ej. "pide el 92% de los martes") sobre una ventana móvil de 8-12 semanas.
- Regla de alerta: `P_empírica(pide este día de semana) > umbral` y no pidió → candidato. Ordenar por esa probabilidad × valor monetario (el M de RFM) para priorizar.
- Calibrar el umbral en el período de validación mirando Precision@15.
- **Verificable:** backtest sobre jun-jul 2026: para cada día de test, la lista top-15 que el sistema hubiera generado y su precisión real. Una tabla `fecha | precision@15` con su promedio.

### Fase 4 — Feature engineering + modelo ML (3-5 días)
- Panel cliente-día (Decisión 3) con features tipo: días desde último pedido, frecuencia en ventanas de 7/14/28/56 días, tasa histórica de pedido para ese día de semana, cajas e importe promedio, tendencia (¿viene pidiendo más o menos?), día de semana, día del mes, feriado, tipo de cliente, localidad.
- Todas con `shift(1)` (Decisión 5).
- Modelos: regresión logística (interpretable, rápida) y XGBoost (el candidato fuerte para tabular). Comparar ambos.
- Manejo del desbalance (habrá muchos más 0 que 1): `scale_pos_weight` o class weights; evaluar con AUC-PR además de ROC.
- **Verificable:** mismas métricas y mismo período de test que el baseline, en la misma tabla. Feature importance graficada y comentada (¿el modelo aprendió cosas razonables?).

### Fase 5 — Evaluación comparativa y decisión (1-2 días)
- Tabla final: RFM vs. logística vs. XGBoost en Precision@15, AUC-ROC, AUC-PR sobre el test.
- Análisis de errores: mirar 10 falsos positivos y 10 falsos negativos con nombre y apellido. ¿Son errores razonables (cliente errático) o hay un patrón que falta capturar?
- Decisión escrita: qué versión va a producción y por qué. **Si el RFM empata al ML, elegir RFM es la respuesta correcta** (más simple, más explicable) y contarlo así en el portfolio suma, no resta.
- **Verificable:** sección "Resultados" del README con la tabla y la decisión justificada.

### Fase 6 — Operacionalización (2-3 días)
- Script `src/alerta_diaria.py`: recibe el export del día (parcial o del día anterior según respuesta a la pregunta abierta 2), matchea nombres (Decisión 7), corre el modelo elegido y genera un Excel simple para el comercial: cliente, lugar de entrega, probabilidad, día que suele pedir, cajas promedio, vendedor asignado, teléfono si existe.
- Diseñado para correrse con doble click o un comando a las ~9:30.
- **Verificable:** simulacro completo con el archivo `Pedidos_18-07-2026.xlsx`: correr el pipeline de punta a punta y revisar la lista generada contra tu conocimiento de los clientes.
- Medición del valor real: llevar registro simple de las llamadas hechas y cuántas terminaron en pedido. Ese número ("recuperamos X pedidos/semana") es oro para el portfolio y para tu empresa.

### Fase 7 — Versión portfolio (2-3 días)
- Generar dataset sintético que replique las propiedades estadísticas del real (distribución de frecuencias por segmento, patrones semanales, estacionalidad) sin ningún dato real: nombres tipo `Cliente_0001`, montos escalados, fechas intactas o desplazadas.
- Republicar los notebooks apuntando al dataset sintético y verificar que las conclusiones cualitativas se sostienen.
- README final con: problema, contexto (export de ERP, sin SQL — contalo, es realista), decisiones (este documento es la base), resultados e impacto.
- **Verificable:** checklist de confidencialidad: cero razones sociales, direcciones, vendedores o importes reales en ningún archivo del repo público (incluyendo historial de Git: el repo público se crea de cero).

---

## Conceptos clave del proyecto (para tu aprendizaje)

**Data leakage (fuga de datos):** cuando el modelo entrena con información que no va a existir en el momento de predecir. Ejemplo sutil de este proyecto: si la feature "pedidos en los últimos 7 días" incluye el día que estoy prediciendo, el modelo "adivina" con información del futuro y las métricas mienten. Es el error #1 que separa un proyecto de curso de uno profesional.

**Split temporal:** en datos con tiempo, el test debe ser siempre posterior al entrenamiento, porque así funciona la realidad. Un split aleatorio mezcla pasado y futuro y sobreestima el rendimiento.

**Precision@k:** de las k alertas que emito, cuántas eran correctas. Cuando la salida del modelo es una lista corta de acciones (llamar a 15 clientes), es más honesta que la accuracy global, que en problemas desbalanceados puede ser altísima sin servir para nada (un modelo que dice "nadie pide nunca" acierta el 90% de los cliente-días).

**Desbalance de clases:** la mayoría de los cliente-días son "no pidió". Sin tratamiento, el modelo aprende a decir siempre que no. Se maneja con pesos de clase y métricas adecuadas (AUC-PR).

**Ventana móvil (rolling window):** las features se calculan sobre las últimas N semanas, no sobre todo el histórico, para que el modelo se adapte si un cliente cambió de ritmo.

**Baseline:** todo modelo complejo debe justificarse contra la alternativa simple. Sin baseline no hay forma de saber si el ML aporta o solo agrega complejidad.

---

## Cronograma tentativo (ritmo part-time)

| Semana | Fases |
|---|---|
| 1 | Fase 0 + Fase 1 |
| 2 | Fase 2 |
| 3 | Fase 3 (primer resultado usable en el trabajo) |
| 4-5 | Fase 4 + Fase 5 |
| 6 | Fase 6 (herramienta en uso real) |
| 7 | Fase 7 (versión pública) |

Nota: al final de la semana 3 ya tenés algo que sirve en tu trabajo, aunque el ML no exista todavía. Eso es deliberado: valor temprano, aprendizaje después.
