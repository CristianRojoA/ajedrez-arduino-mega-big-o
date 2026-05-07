"""
Enfrenta el modelo ML vs minimax.
Uso: python test_modelo.py [ruta_modelo.keras] [num_partidas]
     (sin argumentos → modo interactivo con menú)
"""
import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

MODELOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelos_ml')
MAX_MOV = 150


def _elegir_modelo_interactivo():
    modelos = sorted(
        f for f in os.listdir(MODELOS_DIR) if f.endswith('.keras')
    )
    if not modelos:
        print("No se encontraron modelos en:", MODELOS_DIR)
        sys.exit(1)

    print("\n=== MODELOS DISPONIBLES ===")
    for i, nombre in enumerate(modelos, 1):
        print(f"  {i}. {nombre}")
    print()

    while True:
        entrada = input("Selecciona modelo (número o nombre): ").strip()
        if entrada.isdigit():
            idx = int(entrada) - 1
            if 0 <= idx < len(modelos):
                return os.path.join(MODELOS_DIR, modelos[idx])
        else:
            coincide = [m for m in modelos if entrada.lower() in m.lower()]
            if len(coincide) == 1:
                return os.path.join(MODELOS_DIR, coincide[0])
            if len(coincide) > 1:
                print("Ambiguo, coincide con:", ', '.join(coincide))
                continue
        print("Opción no válida, intenta de nuevo.")


def _elegir_partidas_interactivo():
    while True:
        entrada = input("Número de partidas [10]: ").strip()
        if entrada == '':
            return 10
        if entrada.isdigit() and int(entrada) > 0:
            return int(entrada)
        print("Ingresa un número entero positivo.")


# ── Resolución de argumentos ─────────────────────────────────────────────────

if len(sys.argv) > 1:
    MODELO = sys.argv[1]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
else:
    MODELO = _elegir_modelo_interactivo()
    N = _elegir_partidas_interactivo()
    print()

# ── Carga del motor ──────────────────────────────────────────────────────────

import tensor_aprendizaje
from modeloraul import (inicializar_ajedrez, elegir_movimiento, hacer_movimiento,
                        estado_juego)

print(f"Cargando modelo: {os.path.basename(MODELO)}")
motor = tensor_aprendizaje.MotorML()
if not motor.cargar_modelo(MODELO):
    print("ERROR: no se pudo cargar el modelo.")
    sys.exit(1)
print("Modelo cargado.\n")

# ── Torneo ───────────────────────────────────────────────────────────────────

resultados = {'ML': 0, 'MM': 0, 'empate': 0}

for partida in range(1, N + 1):
    tablero  = inicializar_ajedrez()
    turno    = 0
    num_mov  = 0
    ml_turno = 0 if partida % 2 == 1 else 1

    while True:
        est = estado_juego(tablero, turno)
        if est == 'JAQUE_MATE':
            ganador = 'ML' if turno != ml_turno else 'MM'
            resultados[ganador] += 1
            print(f"  Partida {partida:2d}: JAQUE MATE en mov {num_mov} — gana {ganador}")
            break
        if est == 'TABLAS':
            resultados['empate'] += 1
            print(f"  Partida {partida:2d}: TABLAS en mov {num_mov}")
            break
        if num_mov >= MAX_MOV:
            resultados['empate'] += 1
            print(f"  Partida {partida:2d}: CORTE {MAX_MOV} movimientos — empate técnico")
            break

        if turno == ml_turno:
            mov, dist, valor = motor.predecir(tablero, turno, num_mov)
        else:
            mov = elegir_movimiento(tablero, turno, profundidad=2)

        if mov is None:
            resultados['empate'] += 1
            print(f"  Partida {partida:2d}: sin movimiento — empate")
            break

        hacer_movimiento(tablero, mov[0], mov[1])
        turno   = 1 - turno
        num_mov += 1

# ── Resumen ──────────────────────────────────────────────────────────────────

print(f"\n{'='*40}")
print(f"Resultados tras {N} partidas  (ML alterna blancas/negras):")
print(f"  ML gana  : {resultados['ML']}")
print(f"  Minimax  : {resultados['MM']}")
print(f"  Empates  : {resultados['empate']}")
print(f"  Win rate ML: {resultados['ML'] / N * 100:.0f}%")
