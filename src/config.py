"""Configuración central del proyecto.

Todos los scripts y notebooks importan rutas y constantes desde acá,
así nunca hay paths hardcodeados repetidos por todo el código.
"""
from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────
# Raíz del proyecto = carpeta que contiene a src/
ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

# Archivos de entrada esperados (ajustar nombres si cambian los exports)
ARCHIVO_HISTORICO = DATA_RAW / "Informe_de_ventas_diario_2025.xlsx"
HOJA_BASE_VENTA = "baseventa"

# Salida de la Fase 1
VENTAS_PARQUET = DATA_PROCESSED / "ventas.parquet"

# ── Reglas de negocio ────────────────────────────────────────────
# Decisión 2 (refinada en Fase 1): qué comprobantes cuentan como pedido.
# Se excluyen notas de crédito/débito (ajustes) y también FACTURA 0 /
# NOTA CREDITO 0: son mercadería sin cargo (importe $0, muestras y
# bonificaciones impulsadas por el vendedor), no pedidos del cliente.
COMPROBANTES_VENTA = {
    "Factura",
    "Factura de Crédito Mi Pyme",
    "Factura de Exportación",
}

# Decisión 4: cliente activo si su silencio actual no supera
# K veces su intervalo típico entre pedidos (se calibra en Fase 2/3)
FACTOR_INACTIVIDAD_K = 3.0
TOPE_INACTIVIDAD_DIAS = 60

# Feriados: país base + archivo de excepciones propias
PAIS_FERIADOS = "AR"
FERIADOS_EXTRA_CSV = ROOT / "data" / "feriados_extra.csv"
