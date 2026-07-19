"""Fase 4a — Construcción del panel cliente-día con features.

Transforma los pedidos en un panel de observaciones (punto_entrega, día
operativo) con target binario (¿pidió ese día?) y features calculadas
exclusivamente con información anterior al día objetivo (shift(1) en
todas las ventanas: garantía anti-leakage).

Uso:
    python src/features.py     # lee ventas.parquet, escribe panel.parquet
"""
import pandas as pd

try:
    import config
    from baseline import construir_pedidos
except ImportError:
    from src import config
    from src.baseline import construir_pedidos

MIN_PEDIDOS_DIA_OPERATIVO = 50   # un día con menos pedidos globales que esto
                                 # es feriado/no operativo y sale del panel
WARMUP_DIAS = 28                 # días tras el primer pedido antes de entrar
SALIDA_DIAS = 60                 # días tras el último pedido antes de salir

FEATURES = ["tasa_dia_semana", "freq_2sem", "freq_4sem", "freq_12sem",
            "dias_sin_pedir", "ratio_ausencia", "tendencia", "cajas_prom",
            "dia_semana", "dia_mes"]


def construir_panel(ped: pd.DataFrame, fecha_prediccion=None) -> pd.DataFrame:
    """fecha_prediccion: en producción, fecha futura (la entrega de mañana)
    que se agrega al calendario para generar sus features aunque el
    histórico aún no la contenga. Su target queda en 0 y no se usa."""
    # ── Calendario operativo (excluye domingos y feriados por datos) ──
    actividad = ped.groupby("Fecha").size()
    cal = pd.DatetimeIndex(
        actividad[actividad > MIN_PEDIDOS_DIA_OPERATIVO].index).sort_values()
    if fecha_prediccion is not None:
        fp = pd.Timestamp(fecha_prediccion)
        ped = ped[ped["Fecha"] < fp]          # nada del día objetivo ni después
        cal = cal[cal < fp].append(pd.DatetimeIndex([fp]))

    # ── Grilla punto × día, acotada a la vida de cada punto ──
    rango = ped.groupby("punto_entrega")["Fecha"].agg(["min", "max"])
    puntos, fechas = [], []
    for punto, (fmin, fmax) in rango.iterrows():
        mask = ((cal >= fmin + pd.Timedelta(days=WARMUP_DIAS))
                & (cal <= min(fmax + pd.Timedelta(days=SALIDA_DIAS), cal.max())))
        f = cal[mask]
        puntos.extend([punto] * len(f))
        fechas.extend(f)
    panel = pd.DataFrame({"punto_entrega": puntos, "Fecha": fechas})

    # ── Target ──
    idx_pedidos = ped.set_index(["punto_entrega", "Fecha"]).index
    panel["target"] = (panel.set_index(["punto_entrega", "Fecha"])
                            .index.isin(idx_pedidos).astype(int))
    panel["dia_semana"] = panel["Fecha"].dt.dayofweek
    panel = panel.sort_values(["punto_entrega", "Fecha"]).reset_index(drop=True)

    # ── Features (todas shift(1): solo pasado) ──
    g = panel.groupby("punto_entrega", group_keys=False)

    # La estrella según el EDA: tasa de pedido en ese día de semana,
    # últimas 12 ocurrencias de ese día (~12 semanas)
    gd = panel.groupby(["punto_entrega", "dia_semana"], group_keys=False)
    panel["tasa_dia_semana"] = gd["target"].apply(
        lambda s: s.shift(1).rolling(12, min_periods=4).mean())

    # Frecuencia general en ~2/4/12 semanas de calendario operativo
    for n, nombre in [(12, "freq_2sem"), (24, "freq_4sem"), (72, "freq_12sem")]:
        panel[nombre] = g["target"].apply(
            lambda s: s.shift(1).rolling(n, min_periods=6).mean())

    # Ausencia relativa al propio ritmo (Decisión 4 hecha feature)
    panel["pos"] = g.cumcount()
    pos_pedido = panel["pos"].where(panel["target"] == 1)
    panel["pos_ultimo"] = pos_pedido.groupby(
        panel["punto_entrega"]).transform(lambda s: s.shift(1).ffill())
    panel["dias_sin_pedir"] = panel["pos"] - panel["pos_ultimo"]
    panel["intervalo_tipico"] = 1 / panel["freq_12sem"].clip(lower=1 / 72)
    panel["ratio_ausencia"] = panel["dias_sin_pedir"] / panel["intervalo_tipico"]

    # Tendencia: ¿viene acelerando o enfriándose?
    panel["tendencia"] = panel["freq_4sem"] / panel["freq_12sem"].clip(lower=0.01)

    # Valor típico del pedido
    panel = panel.merge(ped[["punto_entrega", "Fecha", "cajas"]],
                        on=["punto_entrega", "Fecha"], how="left")
    panel["cajas_prom"] = (panel.groupby("punto_entrega", group_keys=False)["cajas"]
                                .apply(lambda s: s.shift(1).rolling(72, min_periods=1).mean()))

    panel["dia_mes"] = panel["Fecha"].dt.day

    # Filas con features completas (el resto es warmup)
    panel = panel.dropna(subset=["tasa_dia_semana", "freq_12sem", "dias_sin_pedir"])
    return panel[["punto_entrega", "Fecha", "target"] + FEATURES]


def main():
    df = pd.read_parquet(config.VENTAS_PARQUET)
    ped = construir_pedidos(df)
    panel = construir_panel(ped)
    salida = config.DATA_PROCESSED / "panel.parquet"
    panel.to_parquet(salida, index=False)
    print(f"Panel: {len(panel):,} filas | tasa target: {panel['target'].mean():.1%}")
    print(f"Guardado: {salida}")


if __name__ == "__main__":
    main()
