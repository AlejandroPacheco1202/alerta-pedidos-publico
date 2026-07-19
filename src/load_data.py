"""Fase 1 — Ingesta y limpieza del histórico de ventas.

Lee la hoja 'base venta' del export del ERP, aplica la limpieza y los
filtros de negocio documentados, y guarda el resultado en parquet.

Uso:
    python src/load_data.py                     # usa la ruta de config.py
    python src/load_data.py ruta/al/export.xlsx # ruta explícita

Cada paso imprime un reporte de validación: filas antes/después y el
porqué de cada filtro. Nada se descarta en silencio.
"""
import sys
import unicodedata

import pandas as pd

try:
    import config  # ejecutado como script desde la raíz: python src/load_data.py
except ImportError:  # importado como módulo: from src import load_data
    from src import config


# ── Normalización de texto ───────────────────────────────────────

def normalizar_texto(s: pd.Series) -> pd.Series:
    """Mayúsculas, sin tildes, sin puntuación redundante, espacios colapsados.

    'Café  Martínez  S.R.L. ' -> 'CAFE MARTINEZ SRL'
    Esto hace comparables los nombres entre el histórico y el reporte
    diario de pedidos (Decisión 7).
    """
    def _norm(x):
        if pd.isna(x):
            return x
        x = str(x).upper().strip()
        # quitar tildes/diacríticos
        x = unicodedata.normalize("NFKD", x)
        x = "".join(c for c in x if not unicodedata.combining(c))
        # puntuación típica de razones sociales
        for ch in [".", ",", "-", "´", "'", '"']:
            x = x.replace(ch, " ")
        # colapsar espacios múltiples
        x = " ".join(x.split())
        return x

    return s.map(_norm)


# ── Pipeline principal ───────────────────────────────────────────

def cargar_y_limpiar(ruta_excel=None, verbose: bool = True) -> pd.DataFrame:
    ruta = ruta_excel or config.ARCHIVO_HISTORICO

    def log(msg):
        if verbose:
            print(msg)

    log(f"Leyendo {ruta} (hoja '{config.HOJA_BASE_VENTA}')...")
    df = pd.read_excel(ruta, sheet_name=config.HOJA_BASE_VENTA)
    n0 = len(df)
    log(f"  Filas leídas: {n0:,}")
    log(f"  Rango de fechas: {df['Fecha'].min():%d/%m/%Y} -> {df['Fecha'].max():%d/%m/%Y}")

    # 1) Tipado -----------------------------------------------------
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    for col in ["Cajas", "Importe Neto", "Importe Descuento", "Bonificación"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    fechas_invalidas = df["Fecha"].isna().sum()
    assert fechas_invalidas == 0, f"{fechas_invalidas} fechas inválidas: revisar export"

    # 2) Filtro: lista de precios DEMO ------------------------------
    # Hallazgo Fase 1: las filas con razón social nula son en un 96%
    # lista 'DEMO' + comprobante 'FACTURA 0' con importe $0: muestras
    # gratis que los vendedores entregan a prospectos. No son pedidos
    # de clientes y no deben entrenar el modelo.
    es_demo = df["Lista Precios"] == "DEMO"
    log(f"\n[Filtro 1] Lista DEMO (muestras a prospectos): -{es_demo.sum():,} filas")
    df = df[~es_demo]

    # 3) Filtro: comprobantes que no son pedido (Decisión 2 refinada)
    # - Notas de crédito/débito: ajustes contables, no pedidos.
    # - FACTURA 0 / NOTA CREDITO 0: mercadería sin cargo (importe ~$0)
    #   impulsada por el vendedor, no cargada por el cliente. Solo el
    #   0.8% de los días-cliente existían únicamente por facturas 0.
    es_venta = df["Tipo de comprobante"].isin(config.COMPROBANTES_VENTA)
    detalle = (
        df.loc[~es_venta, "Tipo de comprobante"].value_counts().to_dict()
    )
    log(f"[Filtro 2] Comprobantes no-pedido: -{(~es_venta).sum():,} filas -> {detalle}")
    df = df[es_venta]

    # 4) Filtro: filas sin identificación de cliente ----------------
    # Tras los filtros anteriores debería quedar casi nada.
    sin_cliente = df["Cliente - Razón Social"].isna() | df["Nombre Fantasía"].isna()
    if sin_cliente.sum():
        log(f"[Filtro 3] Sin razón social / fantasía tras filtros: -{sin_cliente.sum():,} filas")
        df = df[~sin_cliente]

    # 5) Cantidades negativas o nulas en comprobantes de venta ------
    # Una factura con cajas <= 0 es una anomalía a vigilar, no un pedido.
    cajas_raras = df["Cajas"] <= 0
    if cajas_raras.sum():
        log(f"[Filtro 4] Facturas con Cajas <= 0 (anomalías): -{cajas_raras.sum():,} filas")
        df = df[~cajas_raras]

    # 6) Normalización de nombres (Decisión 7) ----------------------
    df = df.copy()
    df["cliente_norm"] = normalizar_texto(df["Cliente - Razón Social"])
    df["fantasia_norm"] = normalizar_texto(df["Nombre Fantasía"])
    # Clave de punto de entrega (Decisión 1): fantasía + dirección,
    # porque un mismo nombre de fantasía puede repetirse entre clientes.
    df["punto_entrega"] = (
        df["fantasia_norm"] + " | " + normalizar_texto(df["Dirección de Entrega"])
    )

    # 7) Columnas auxiliares de calendario --------------------------
    df["dia_semana"] = df["Fecha"].dt.dayofweek  # 0=lunes ... 6=domingo

    # ── Reporte final de validación ────────────────────────────────
    log("\n── Validación final " + "─" * 40)
    log(f"  Filas: {n0:,} -> {len(df):,} ({len(df)/n0:.1%} conservado)")
    log(f"  Rango de fechas: {df['Fecha'].min():%d/%m/%Y} -> {df['Fecha'].max():%d/%m/%Y}")
    log(f"  Clientes (razón social): {df['cliente_norm'].nunique():,}")
    log(f"  Puntos de entrega: {df['punto_entrega'].nunique():,}")
    log(f"  Días-punto de entrega (candidato a 'pedido'): "
        f"{df.groupby(['punto_entrega', 'Fecha']).ngroups:,}")
    log(f"  Nulos restantes por columna clave: "
        f"{df[['Fecha', 'cliente_norm', 'punto_entrega', 'Cajas']].isna().sum().sum()}")

    return df


def main():
    ruta = sys.argv[1] if len(sys.argv) > 1 else None
    df = cargar_y_limpiar(ruta)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.VENTAS_PARQUET, index=False)
    print(f"\nGuardado: {config.VENTAS_PARQUET} "
          f"({config.VENTAS_PARQUET.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
