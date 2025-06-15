# 🧠 Detección de Lavado de Activos con Aprendizaje por Refuerzo

Este proyecto implementa un sistema basado en aprendizaje por refuerzo (Reinforcement Learning, RL) para detectar posibles actividades relacionadas con el lavado de activos, usando el comportamiento transaccional de los clientes en los últimos seis meses. El modelo se entrena sobre datos históricos de Reportes de Operaciones Sospechosas (ROS) y se retroalimenta mensualmente con nuevos casos confirmados y feedback del equipo de investigaciones.

## 🎯 Objetivo

Desarrollar un agente inteligente que aprenda a clasificar actividades sospechosas en clientes bancarios, maximizando la detección temprana de operaciones relacionadas con lavado de activos, y minimizando falsos positivos, mediante retroalimentación continua.

## 🧩 Componentes principales

- **Entorno Gymnasium personalizado:** Simula los clientes, sus transacciones y la evaluación de alertas generadas.
- **Agente RL:** Entrenado con algoritmos como PPO o DQN usando `stable-baselines3`.
- **Función de recompensa adaptativa:** Se ajusta con el feedback del equipo de investigaciones.
- **Actualización mensual:** Incorporación de nuevos casos confirmados y ajustes en el modelo.

## 📁 Estructura del proyecto

```

reinforcement-aml/
│
├── data/
│   ├── raw/              # Datos históricos de ROS y actividades
│   └── processed/        # Datos listos para el entorno Gymnasium
│
├── models/
│   └── train\_agent.py    # Script para entrenar el agente RL
│
├── feedback/
│   └── feedback\_update.py # Módulo para incorporar feedback mensual
│
├── notebooks/
│   └── exploracion.ipynb  # Análisis exploratorio y visualizaciones
│
├── utils/
│   └── preprocessing.py  # Transformaciones y extracción de features
|
├── src/
│   └── app.py  # app principal para ejecutar el entorno y el agente, y revisar las alertas para generar la data para generar el feedback del modelo
│
├── requirements.txt
└── README.md

```

## 🛠 Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu_usuario/reinforcement-aml.git
cd reinforcement-aml
````

2. Crear y activar entorno virtual:

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

## 🧪 Ejecución rápida

Entrenar el agente con datos simulados:

```bash
python models/train_agent.py
```

Evaluar el rendimiento del agente:

```bash
python models/evaluate_agent.py
```

## 🔁 Retroalimentación mensual

Cada mes, el equipo de investigaciones proveerá:

* Confirmaciones de ROS acertados
* Nuevos casos detectados manualmente
* Revisión de alertas falsas

El módulo `feedback/feedback_update.py` permite incorporar estos casos para reentrenar o ajustar el modelo.

## 🧠 Tecnologías utilizadas

* [Gymnasium](https://gymnasium.farama.org/)
* [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
* [scikit-learn](https://scikit-learn.org/)
* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* Python 3.10+

## 🧭 Futuras extensiones

* Incluir múltiples tipos de acciones (alerta baja, media, alta).
* Entrenamiento continuo (*online learning*) en producción.
* Dashboard en tiempo real para analistas.
* Métricas de explicabilidad de decisiones del agente.

## 🏦 Contexto del proyecto

Este proyecto se desarrolla en el área de Cumplimiento del Banco de Crédito del Perú (BCP), con el objetivo de fortalecer las herramientas analíticas para detección de lavado de activos mediante técnicas modernas de inteligencia artificial.

## 👤 Autor

**César Aaron Fernández Niño**
Data Scientist en Cumplimiento – BCP Perú
Email: \[[Cesar.FernandezNino@colorado.edu](Cesar.FernandezNino@colorado.edu)]
LinkedIn: [https://www.linkedin.com/in/c%C3%A9sar-aar%C3%B3n-fern%C3%A1ndez-ni%C3%B1o-296182116/](https://www.linkedin.com/in/c%C3%A9sar-aar%C3%B3n-fern%C3%A1ndez-ni%C3%B1o-296182116/)

---

> "La clave no es reemplazar al investigador humano, sino empoderarlo con herramientas inteligentes que aprendan de él."


## 📄 Licencia
MIT License

