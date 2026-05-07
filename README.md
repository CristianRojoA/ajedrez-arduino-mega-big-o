# ♟️ Jaque Mate al Big O: Motor de Ajedrez Autónomo con ML

> **Asignatura:** Diseño y Análisis de Algoritmos · Universidad de La Serena  
> **Hardware:** Raspberry Pi 5 (4 GB RAM) · **Interfaz:** Monitor Touch 10.1″ vía HDMI/DSI  
> **Paradigma de IA:** Aprendizaje Supervisado — Red Neuronal Residual (inspirada en AlphaZero)  
> **SRS:** ISO/IEC/IEEE 29148:2018 · Versión 3.2

---

## 👥 Integrantes y Roles (Grupo 1)

| Rol | Nombre | Responsabilidad |
|-----|--------|----------------|
| **Project Manager (PM)** | Cristian Rojo | Coordinación, gestión de hitos y documentación SRS |
| **Arquitecto (ARCH)** | Mauricio Hernández | Diseño del pipeline ML, arquitectura de red neuronal |
| **Arquitecto (ARCH)** | Ignacio Brizuela | Integración hardware, configuración Raspberry Pi OS |
| **Desarrollador (DEV)** | Francisco Seura | Motor lógico FIDE, interfaz gráfica Kivy |
| **Desarrollador (DEV)** | Raúl Muñoz | Entrenamiento del modelo, encoding de tensores |

---

## 🎯 Objetivo del Proyecto

Demostrar empíricamente la **Notación Asintótica (Big-O)** de un motor de ajedrez autónomo ejecutado íntegramente en una Raspberry Pi 5. El sistema ejecuta partidas máquina vs. máquina mientras registra en microsegundos los tiempos de cada fase del pipeline de IA, evidenciando:

- **O(n)** — Generación de movimientos legales FIDE y encoding del tablero a tensor
- **O(1)** — Inferencia de la red neuronal (el costo es constante dado que la arquitectura del modelo es fija)

---

## 🧠 Arquitectura de Inteligencia Artificial

El motor utiliza **Aprendizaje Supervisado** entrenado sobre partidas en formato PGN. La arquitectura es una red neuronal residual dual inspirada en AlphaZero/LeelaChessZero:

```
Entrada: tensor (8 × 8 × 14)
  ├── Planos 0–5  : piezas blancas (peón, caballo, alfil, torre, reina, rey)
  ├── Planos 6–11 : piezas negras
  ├── Plano 12    : turno activo
  └── Plano 13    : número de movimiento normalizado

Torre residual: 4 bloques Conv2D (128 filtros, kernel 3×3) + BatchNorm + ReLU + skip connections

Cabezas de salida:
  ├── Política  → 4096 probabilidades (64×64 posibles movimientos desde×hasta)
  └── Valor     → escalar ∈ [-1, +1] (evaluación de la posición)
```

**Configuración por defecto** (`model_config.yaml`):

| Parámetro | Valor |
|-----------|-------|
| Filtros | 128 |
| Bloques residuales | 4 |
| Dropout | 0.3 |
| Batch size | 256 |
| Epochs | 50 |
| Learning rate | 0.001 |
| Paciencia (early stopping) | 8 |
| Validación | 15% |
| Tamaño de política | 4096 |

---

## 🛠️ Stack Tecnológico

| Componente | Tecnología |
|------------|-----------|
| **Hardware** | Raspberry Pi 5 (4 GB RAM LPDDR4X) |
| **Sistema Operativo** | Raspberry Pi OS Lite (64-bit, ARM64) |
| **Lenguaje** | Python 3.11+ |
| **Interfaz gráfica** | Kivy (resolución 1520×960 → monitor 1920×1200) |
| **Motor lógico** | `modeloraul.py` — arreglo unidimensional de 64 enteros |
| **ML (entrenamiento)** | TensorFlow / Keras |
| **ML (inferencia)** | TensorFlow Lite u ONNX Runtime |
| **Formato de partidas** | PGN (`python-chess`) |
| **Métricas de tiempo** | `time.perf_counter()` (resolución de microsegundos) |
| **Pantalla** | Monitor IPS Touch 10.1″ vía HDMI/micro-HDMI (o DSI) |

---

## 📁 Estructura del Repositorio

```
ajedrez-raspberry-pi-5-main/
│
├── vistafrancisco.py          # Punto de entrada — ChessApp (Kivy ScreenManager)
├── vista_config.py            # Constantes de UI, rutas, skins, resolución
├── vista_screens_inicio.py    # Pantallas: selección de modo (Minimax vs ML)
├── vista_screens_juego.py     # Pantallas: menú, video, partidas, juego
├── vista_screens_ml.py        # Pantallas: EntrenarScreen, ProbarModeloScreen
├── vista_paneles.py           # Paneles laterales: historial de movimientos, chat
├── vista_tablero.py           # Widget del tablero con sprites y animaciones
│
├── modeloraul.py              # Motor lógico FIDE: generador de movimientos,
│                              #   Minimax/alfa-beta (modo legacy), hash, evaluación
├── tensor_aprendizaje.py      # MotorML: encoding (8×8×14), entrenamiento y
│                              #   predicción con la red neuronal residual
├── partidas_pgn.py            # Exportación/importación PGN, notación SAN
├── test_modelo.py             # Enfrenta modelo ML vs Minimax y reporta win rate
│
├── modelos_ml/                # Modelos entrenados (.keras) y metadatos (.json)
│   ├── motor_ml_YYYYMMDD_HHMMSS.keras
│   └── motor_ml_YYYYMMDD_HHMMSS_meta.json
│
├── assets/                    # Sprites de piezas (3 skins disponibles)
│   ├── clasico/               # Estilo clásico (PNG de alta resolución)
│   ├── shield/                # Estilo shield (PNG ligero)
│   └── vocaloid_backup/       # Estilo alternativo
│
├── partidas/
│   └── partidas aprendizaje/  # Archivos .pgn para entrenamiento supervisado
│
├── data/                      # Logs de tiempos de inferencia (.csv) generados en ejecución
├── docs/                      # Documentación técnica y diagramas
│
├── model_config.yaml          # Configuración del modelo ML (auto-generado si no existe)
├── changelog.md               # Historial de cambios del proyecto
└── README.md                  # Este archivo
```

---

## ⚙️ Instalación y Puesta en Marcha

### 1. Requisitos previos

```bash
# Sistema operativo: Raspberry Pi OS Lite 64-bit
# Python 3.11 o superior

sudo apt update && sudo apt install -y python3-pip python3-venv
```

### 2. Clonar el repositorio

```bash
git clone https://github.com/<usuario>/ajedrez-raspberry-pi-5.git
cd ajedrez-raspberry-pi-5
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

**`requirements.txt` esperado:**
```
tensorflow>=2.16
onnxruntime
kivy>=2.3
python-chess
numpy
pyyaml
```

### 4. Configurar la pantalla táctil (HDMI/DSI)

El monitor Touch 10.1″ debe estar conectado al puerto micro-HDMI 0 o al puerto DSI de la Raspberry Pi 5. No se requieren drivers adicionales en Raspberry Pi OS 64-bit.

### 5. Entrenar un modelo (opcional — se incluyen modelos pre-entrenados)

```bash
# Desde la interfaz gráfica: seleccionar modo ML → Entrenar Modelo
# O desde consola:
python tensor_aprendizaje.py
```

### 6. Ejecutar el sistema

```bash
# Con prioridad máxima para proteger las métricas Big-O
sudo nice -n -20 python vistafrancisco.py
```

---

## 🚀 Modos de Operación

El sistema ofrece dos modos seleccionables desde la pantalla de inicio:

| Modo | Descripción |
|------|-------------|
| **Minimax** | Motor clásico con poda alfa-beta. Profundidad configurable. Útil para comparación baseline. |
| **ML** | Motor de red neuronal residual. Inferencia O(1). Incluye submenú para entrenar y probar modelos. |

### Pantallas disponibles (Kivy ScreenManager)

```
ModeScreen      → Selección de modo (Minimax / ML)
  ├── MLScreen  → Menú ML (Entrenar / Probar / Jugar)
  │     ├── EntrenarScreen   → Carga PGNs, entrena y guarda modelo
  │     └── ProbarModeloScreen → ML vs Minimax, reporte de win rate
  └── MenuScreen → Configuración de skin y opciones de partida
        ├── VideoScreen      → Intro animada
        ├── PartidasScreen   → Historial de partidas guardadas
        └── GameScreen       → Partida en curso (tablero + paneles)
```

---

## 📊 Métricas y Análisis Big-O

El sistema registra automáticamente los tiempos de cada fase en `data/`:

```
data/
└── metricas_YYYYMMDD_HHMMSS.csv
    ├── turno          : número de jugada
    ├── fase           : "encoding" | "inferencia" | "filtrado_legal"
    ├── tiempo_us      : tiempo en microsegundos (time.perf_counter)
    └── movimientos_n  : cantidad de movimientos legales generados (para O(n))
```

**Comportamiento esperado en las gráficas:**

- La curva de **encoding** y **generación de movimientos** crece linealmente → **O(n)**
- La curva de **inferencia del modelo** permanece constante en todos los turnos → **O(1)**

---

## 🧪 Test: Modelo ML vs Minimax

```bash
# 10 partidas (ML alterna blancas/negras)
python test_modelo.py modelos_ml/motor_ml_20260503_215250.keras 10
```

Salida esperada:
```
Cargando modelo: motor_ml_20260503_215250.keras
Modelo cargado.

  Partida  1: JAQUE MATE en mov 47 — gana ML
  ...
========================================
Resultados tras 10 partidas (ML alterna blancas/negras):
  ML gana  : 7
  Minimax  : 2
  Empates  : 1
  Win rate ML: 70%
```

---

## 🗓️ Hitos del Proyecto

| Hito | Fecha | Descripción |
|------|-------|-------------|
| **Hito 1** | 07/04/2026 | Aprobación del SRS v3.0 |
| **Hito 2** | 30/04/2026 | Motor lógico FIDE completo + interfaz Kivy operativa |
| **Hito 3** | 28/05/2026 | Auditoría QA: pruebas de estrés + extracción de logs CSV |
| **Hito 4** | 25/06/2026 | Defensa final — demostración física ante la comisión |

---

## 📋 Estándar de Documentación

Este proyecto se documenta bajo el estándar **ISO/IEC/IEEE 29148:2018**. El SRS completo (v3.2) está disponible en `/docs/`.

---

## 📄 Licencia

MIT License — ver `LICENSE` para detalles.
