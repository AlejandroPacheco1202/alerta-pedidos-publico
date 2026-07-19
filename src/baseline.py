"""Fase 3 — Baseline de reglas: perfil semanal + regla de actividad.

Para cada punto de entrega calcula, con una ventana móvil de 12 semanas,
la probabilidad empírica de que pida cada día de la semana. La alerta del
día D lista los puntos activos con P(pide el día-de-semana de D) >= umbral
que no aparecen en el reporte de pedidos cargados.

Uso:
    python src/baseline.py backtest
        Corre el backtest en validación (abr-may 2026) y test (jun-jul 2026).

    python src/baseline.py alerta 2026-07-18 data/raw/Pedidos_18-07-2026.xlsx
        Genera outputs/alerta_2026-07-18.xlsx con la lista de llamadas.

Hallazgos incorporados:
    - Feriados: en días no laborables no hay pedidos (verificado: 15/06/2026,
      0 pedidos en toda la empresa). La alerta no corre esos días.
    - Matching reporte diario -> histórico: por nombre de fantasía
      normalizado, 95% de match exacto verificado con el reporte del 18/07.
"""
import sys

import pandas as pd

try:
    import config
    from load_data import normalizar_texto
except ImportError:
    from src import config
    from src.load_data import normalizar_texto

VENTANA_DIAS = 84          # 12 semanas
UMBRAL_ALERTA = 0.6        # calibrado en validación abr-may 2026
MIN_PEDIDOS_VENTANA = 4    # menos que esto = sin señal suficiente


# ── Calendario ───────────────────────────────────────────────────

def es_dia_de_alerta(fecha) -> bool:
    """Días con reparto: lunes a sábado, excluyendo feriados nacionales
    y los feriados propios listados en data/feriados_extra.csv."""
    f = pd.Timestamp(fecha)
    if f.dayofweek == 6:  # domingo
        return False
    import holidays
    feriados = holidays.country_holidays(config.PAIS_FERIADOS, years=f.year)
    if f.date() in feriados:
        return False
    if config.FERIADOS_EXTRA_CSV.exists():
        extra = pd.read_csv(config.FERIADOS_EXTRA_CSV, parse_dates=["fecha"])
        if f.normalize() in set(extra["fecha"]):
            return False
    return True


# ── Núcleo del baseline ──────────────────────────────────────────

def construir_pedidos(df: pd.DataFrame) -> pd.DataFrame:
    """Líneas de factura -> pedidos (punto_entrega + Fecha)."""
    ped = (df.groupby(["punto_entrega", "Fecha"], as_index=False)
             .agg(cajas=("Cajas", "sum"),
                  importe=("Importe Neto", "sum"),
                  cliente=("cliente_norm", "first")))
    ped["dia_semana"] = ped["Fecha"].dt.dayofweek
    return ped.sort_values(["punto_entrega", "Fecha"])


def perfil_a_fecha(ped: pd.DataFrame, fecha_corte) -> pd.DataFrame:
    """Perfil de cada punto usando SOLO pedidos con Fecha < fecha_corte.

    Este '<' estricto es la garantía anti-leakage: el perfil que se usa
    para predecir el día D jamás ve el día D.
    """
    fc = pd.Timestamp(fecha_corte)
    win = ped[(ped["Fecha"] < fc)
              & (ped["Fecha"] >= fc - pd.Timedelta(days=VENTANA_DIAS))]
    win = win.sort_values(["punto_entrega", "Fecha"])

    # cuántas veces ocurrió cada día de semana dentro de la ventana
    dias = pd.date_range(fc - pd.Timedelta(days=VENTANA_DIAS),
                         fc - pd.Timedelta(days=1))
    ocurrencias = pd.Series(dias.dayofweek).value_counts().to_dict()

    prof = win.groupby("punto_entrega").agg(
        n_ventana=("Fecha", "count"), ultimo=("Fecha", "max"),
        cajas_prom=("cajas", "mean"), importe_prom=("importe", "mean"),
        cliente=("cliente", "first"))

    conteo = (win.groupby(["punto_entrega", "dia_semana"]).size()
                 .unstack(fill_value=0))
    for d in range(6):
        if d in conteo:
            prof[f"p_dia{d}"] = (conteo[d] / ocurrencias[d]).clip(upper=1)
        else:
            prof[f"p_dia{d}"] = 0.0
    prof = prof.fillna({f"p_dia{d}": 0.0 for d in range(6)})

    # Decisión 4: activo si su silencio no supera K veces su ritmo típico
    prof["intervalo_med"] = (win.groupby("punto_entrega")["Fecha"].diff()
                                .dt.days.groupby(win["punto_entrega"]).median())
    prof["dias_sin_pedir"] = (fc - prof["ultimo"]).dt.days
    tope = (config.FACTOR_INACTIVIDAD_K * prof["intervalo_med"]).clip(
        upper=config.TOPE_INACTIVIDAD_DIAS)
    prof["activo"] = ((prof["dias_sin_pedir"] <= tope)
                      & (prof["n_ventana"] >= MIN_PEDIDOS_VENTANA))
    return prof


def generar_alerta(ped, fecha, lugares_ya_pidieron: set) -> pd.DataFrame:
    """Lista priorizada de puntos que 'deberían' pedir hoy y no aparecen."""
    f = pd.Timestamp(fecha)
    prof = perfil_a_fecha(ped, f)
    act = prof[prof["activo"]].copy()
    act["p_hoy"] = act[f"p_dia{f.dayofweek}"]
    act["fantasia"] = act.index.str.split(" | ", regex=False).str[0]

    alerta = act[(act["p_hoy"] >= UMBRAL_ALERTA)
                 & (~act["fantasia"].isin(lugares_ya_pidieron))]
    return (alerta.sort_values(["p_hoy", "importe_prom"], ascending=False)
                  [["cliente", "fantasia", "p_hoy", "dias_sin_pedir",
                    "intervalo_med", "cajas_prom", "importe_prom"]])


def leer_reporte_diario(ruta) -> set:
    """Lee el reporte de pedidos cargados y devuelve lugares normalizados."""
    rep = pd.read_excel(ruta)
    lugares = normalizar_texto(rep["Lugar de Entrega"]).dropna().unique()
    return set(lugares)


# ── Backtest ─────────────────────────────────────────────────────

def backtest(ped, desde, hasta, umbral=UMBRAL_ALERTA, cada_n_dias=1):
    """Para cada día hábil del rango: ¿los que el baseline rankea arriba
    efectivamente pidieron? (precision@15 y precisión del set alertado)."""
    filas = []
    fechas = [f for f in pd.date_range(desde, hasta) if es_dia_de_alerta(f)]
    for fc in fechas[::cada_n_dias]:
        prof = perfil_a_fecha(ped, fc)
        act = prof[prof["activo"]].copy()
        act["p_hoy"] = act[f"p_dia{fc.dayofweek}"]
        pidieron = set(ped.loc[ped["Fecha"] == fc, "punto_entrega"])
        rank = act.sort_values("p_hoy", ascending=False)
        flag = act[act["p_hoy"] >= umbral]
        filas.append({
            "fecha": fc,
            "p@15": rank.head(15).index.isin(pidieron).mean(),
            "n_alertados": len(flag),
            "precision_alertados": (flag.index.isin(pidieron).mean()
                                    if len(flag) else float("nan")),
        })
    return pd.DataFrame(filas)


# ── CLI ──────────────────────────────────────────────────────────

def main():
    modo = sys.argv[1] if len(sys.argv) > 1 else "backtest"
    df = pd.read_parquet(config.VENTAS_PARQUET)
    ped = construir_pedidos(df)

    if modo == "backtest":
        print("Validación (abr-may 2026):")
        val = backtest(ped, "2026-04-01", "2026-05-31")
        print(f"  precision@15: {val['p@15'].mean():.1%} | "
              f"alertados/día: {val['n_alertados'].mean():.0f} | "
              f"precisión: {val['precision_alertados'].mean():.1%}")
        print("Test (jun - 18 jul 2026):")
        test = backtest(ped, "2026-06-01", "2026-07-18")
        print(f"  precision@15: {test['p@15'].mean():.1%} | "
              f"alertados/día: {test['n_alertados'].mean():.0f} | "
              f"precisión: {test['precision_alertados'].mean():.1%}")
        config.OUTPUTS.mkdir(exist_ok=True)
        test.to_csv(config.OUTPUTS / "backtest_baseline_test.csv", index=False)
        print(f"Detalle diario: {config.OUTPUTS / 'backtest_baseline_test.csv'}")

    elif modo == "alerta":
        fecha, ruta_reporte = sys.argv[2], sys.argv[3]
        if not es_dia_de_alerta(fecha):
            print(f"{fecha} no es día de reparto (domingo o feriado): sin alerta.")
            return
        lugares = leer_reporte_diario(ruta_reporte)
        alerta = generar_alerta(ped, fecha, lugares)
        salida = config.OUTPUTS / f"alerta_{fecha}.xlsx"
        config.OUTPUTS.mkdir(exist_ok=True)
        alerta.round(2).to_excel(salida)
        print(f"{len(alerta)} puntos en alerta -> {salida}")
        print(alerta.head(15).round(2).to_string())

    else:
        print(__doc__)


if __name__ == "__main__":
    main()
