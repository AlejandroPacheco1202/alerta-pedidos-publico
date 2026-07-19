"""Fase 7 — Generador de dataset sintético para la versión pública.

Crea un export de ventas ficticio con las mismas propiedades estadísticas
que el dataset real (documentadas en el EDA), sin ningún dato real:

- Dos rutas de reparto (Mar-Jue-Sáb y Lun-Mié-Vie) + clientes mixtos.
- Segmentos de frecuencia: 2-3 veces/semana, semanal, quincenal, esporádico.
- Patrón por día de semana estable, con ruido realista (ausencias ~5-15%).
- Calendario lun-sáb con feriados argentinos.
- Altas y bajas de clientes a lo largo del período.
- Ruido de limpieza deliberado: muestras DEMO sin razón social y notas de
  crédito, para que el pipeline de limpieza tenga trabajo real que mostrar.

Uso:
    python src/generar_sintetico.py
    -> data/raw/Informe_de_ventas_diario_2025.xlsx (sintético)

Todo nombre, dirección e importe es generado; cualquier parecido con la
realidad es estadístico, que es exactamente la idea.
"""
import numpy as np
import pandas as pd

try:
    import config
except ImportError:
    from src import config

RNG = np.random.default_rng(42)

N_PUNTOS = 420
FECHA_INI, FECHA_FIN = "2025-01-02", "2026-07-20"

FERIADOS = pd.to_datetime([
    "2025-03-03", "2025-03-04", "2025-03-24", "2025-04-02", "2025-04-18",
    "2025-05-01", "2025-05-02", "2025-06-16", "2025-06-20", "2025-07-09",
    "2025-08-15", "2025-11-21", "2025-12-08", "2025-12-25",
    "2026-01-01", "2026-02-16", "2026-02-17", "2026-03-24", "2026-04-02",
    "2026-04-03", "2026-05-01", "2026-05-25", "2026-06-15", "2026-07-09",
])

RUBROS = ["Cafetería", "Restaurante", "Hotel", "Almacén", "Supermercado",
          "Catering", "Estación de Servicio"]
ARTICULOS = [("Pan de masa madre x un.", 9500), ("Medialunas x doc.", 12800),
             ("Baguette x un.", 6200), ("Pan de campo x kg", 11400),
             ("Facturas surtidas x doc.", 13900), ("Tostadas x paq.", 7800),
             ("Budín de limón x un.", 15200), ("Pan lactal x un.", 8900),
             ("Criollitos x kg", 9900), ("Chipá x kg", 16800)]
VENDEDORES = [f"Vendedor {c}" for c in "ABCDEFG"]
LOCALIDADES = ["CABA", "Vicente López", "San Isidro", "Pilar", "Morón",
               "Quilmes", "Lomas de Zamora", "Tigre"]


def generar():
    cal = pd.date_range(FECHA_INI, FECHA_FIN)
    cal = cal[(cal.dayofweek < 6) & (~cal.isin(FERIADOS))]

    # ── Población de puntos de entrega ──
    puntos = []
    for i in range(N_PUNTOS):
        seg = RNG.choice(["frecuente", "semanal", "quincenal", "esporadico"],
                         p=[0.15, 0.48, 0.22, 0.15])
        ruta = RNG.choice(["MJS", "LMV", "mixto"], p=[0.52, 0.28, 0.20])
        dias_ruta = {"MJS": [1, 3, 5], "LMV": [0, 2, 4],
                     "mixto": list(range(6))}[ruta]
        # probabilidad de pedido por día de semana según segmento
        p = np.zeros(6)
        if seg == "frecuente":
            for d in dias_ruta[:3]:
                p[d] = RNG.uniform(0.80, 0.97)
        elif seg == "semanal":
            dia = RNG.choice(dias_ruta)
            p[dia] = RNG.uniform(0.70, 0.95)
            if RNG.random() < 0.3:                      # día secundario
                p[RNG.choice(dias_ruta)] = max(p[RNG.choice(dias_ruta)],
                                               RNG.uniform(0.15, 0.35))
        elif seg == "quincenal":
            p[RNG.choice(dias_ruta)] = RNG.uniform(0.35, 0.55)
        else:
            p[dias_ruta] = RNG.uniform(0.03, 0.10)
        # vida del cliente: algunos entran tarde o se dan de baja
        ini = cal[0] if RNG.random() < 0.75 else RNG.choice(cal[:300])
        fin = cal[-1] if RNG.random() < 0.85 else RNG.choice(cal[200:])
        puntos.append({
            "id": i, "p_dia": p, "alta": ini, "baja": fin,
            "razon": f"CLIENTE {i:04d} " + RNG.choice(["SA", "SRL", "SAS"]),
            "fantasia": f"PUNTO {i:04d}",
            "direccion": f"CALLE {RNG.integers(1, 90)} N {RNG.integers(100, 9900)}",
            "localidad": RNG.choice(LOCALIDADES),
            "rubro": RNG.choice(RUBROS),
            "vendedor": RNG.choice(VENDEDORES),
            "tipo_cliente": RNG.choice(["Gastronomía", "Retail", "Corporativo"],
                                       p=[0.6, 0.3, 0.1]),
            "cajas_base": RNG.uniform(1.5, 20),
        })

    # ── Generación de pedidos y líneas ──
    filas = []
    for pt in puntos:
        vida = cal[(cal >= pt["alta"]) & (cal <= pt["baja"])]
        pedido = RNG.random(len(vida)) < pt["p_dia"][vida.dayofweek]
        for fecha in vida[pedido]:
            infl = 1.03 ** ((fecha - cal[0]).days / 30)   # inflación mensual 3%
            n_lineas = RNG.integers(1, 5)
            arts = RNG.choice(len(ARTICULOS), size=n_lineas, replace=False)
            for a in arts:
                nombre, precio = ARTICULOS[a]
                cajas = max(0.5, RNG.normal(pt["cajas_base"] / n_lineas, 1.5))
                filas.append({
                    "Fecha": fecha, "Tipo de Cliente": pt["tipo_cliente"],
                    "Lista Precios": "GENERAL", "Vendedor": pt["vendedor"],
                    "Cliente - Razón Social": pt["razon"],
                    "Nombre Fantasía": pt["fantasia"],
                    "Dirección de Entrega": pt["direccion"],
                    "Localidad": pt["localidad"], "Rubro": pt["rubro"],
                    "Artículo - Desc. Gen.": nombre,
                    "Tipo de comprobante": "Factura",
                    "Cajas": round(cajas, 2),
                    "Importe Neto": round(cajas * precio * infl, 2),
                    "Importe Descuento": 0.0, "Bonificación": 0.0,
                    "Mes": fecha.month,
                })

    df = pd.DataFrame(filas)

    # ── Ruido de limpieza deliberado ──
    # a) Muestras DEMO a prospectos (sin razón social, importe 0)
    demo_fechas = RNG.choice(cal, size=400)
    demo = pd.DataFrame({
        "Fecha": demo_fechas, "Tipo de Cliente": None,
        "Lista Precios": "DEMO",
        "Vendedor": RNG.choice(VENDEDORES, size=400),
        "Cliente - Razón Social": None, "Nombre Fantasía": None,
        "Dirección de Entrega": None,
        "Localidad": RNG.choice(LOCALIDADES, size=400),
        "Rubro": None,
        "Artículo - Desc. Gen.": RNG.choice(
            [f"DEMO {a[0]}" for a in ARTICULOS], size=400),
        "Tipo de comprobante": "FACTURA 0",
        "Cajas": np.round(RNG.uniform(0.5, 2, size=400), 2),
        "Importe Neto": 0.0, "Importe Descuento": 0.0, "Bonificación": 0.0,
        "Mes": pd.DatetimeIndex(demo_fechas).month,
    })
    # b) Notas de crédito (~2% de los pedidos, importe negativo)
    nc = df.sample(frac=0.02, random_state=1).copy()
    nc["Tipo de comprobante"] = "Nota de crédito"
    nc["Importe Neto"] *= -1

    df = (pd.concat([df, demo, nc])
            .sort_values("Fecha").reset_index(drop=True))

    salida = config.DATA_RAW / "Informe_de_ventas_diario_2025.xlsx"
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(salida, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name=config.HOJA_BASE_VENTA, index=False)
    print(f"Dataset sintético: {len(df):,} filas, "
          f"{df['Cliente - Razón Social'].nunique()} clientes -> {salida}")


if __name__ == "__main__":
    generar()
