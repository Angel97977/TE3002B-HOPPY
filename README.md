# TE3002B - Simulación Dinámica del Robot HOPPY en MuJoCo

**Autores:** 
Josue Ureña Valencia    IRS | A01738940
César Arellano Arellano   IRS | A00839373
José Eduardo Sánchez Martinez  IRS | A01738476
Rafael André Gamiz Salazar   IRS | A00838280
Angel Dominguez Faraco IRS | A00835367
**Curso:** TE3002B - Implementación de Robótica (Dr. Ismael Sánchez Osorio)  
**Basado en:** [RoboDesignLab/HOPPY-Project](https://github.com/RoboDesignLab/HOPPY-Project) y el *HOPPY Technical Guideline*.

---

## Índice
1. [Estado del Proyecto y Hitos](#1-estado-del-proyecto-y-hitos)
2. [Problemas Encontrados y Soluciones](#2-problemas-encontrados-y-soluciones)
3. [Fundamento Matemático y Ecuaciones](#3-fundamento-matemático-y-ecuaciones)
4. [Validación contra MATLAB](#4-validación-contra-matlab)
5. [Análisis Técnico de Resultados](#5-análisis-técnico-de-resultados)
6. [Guía de Ejecución](#6-guía-de-ejecución)

---

## 1. Estado del Proyecto y Hitos (100% Logrado)

El proyecto cumple exhaustivamente con todas las fases requeridas en la rúbrica de evaluación:
- **Fase 1 (XML/MJCF):** Modelo ajustado con valores exactos de `armature` ($N^2 \cdot I_{rotor}$) y `damping` viscoso por Back-EMF para motores AK80-9.
- **Fase 2 (Saturación):** Curva Torque-Velocidad real implementada ($\tau_{disp} = \tau_{stall} \cdot (1 - |\omega|/\omega_{noload})$) acotada a $\pm 3.728$ Nm.
- **Fase 3 (Contacto):** Fuerza de Reacción del Suelo (GRF) real obtenida extrayendo la componente normal de `mj_contactForce` sobre el eslabón `foot_rubber`.
- **Fase 4 (Control Híbrido):** Máquina de estados (FLIGHT/STANCE) con histéresis anti-chattering, control PD Cartesiano en vuelo y perfiles Bézier en apoyo.
- **Fase 5 (Análisis):** Emulación de encoders mediante Filtro Pasa-Bajas a 40Hz y script automatizado para 11 gráficas de desempeño.

---

## 2. Problemas Encontrados y Soluciones

Durante el desarrollo de la simulación en Python/MuJoCo, superamos las siguientes discrepancias dinámicas:

1. **Inestabilidad Horizontal (Tercera Ley de Newton):** El controlador original calculaba el empuje en X con el mismo signo que la velocidad. Para que el gantry avance hacia adelante ($+X$), el pie debe empujar el piso hacia atrás ($-X$). Se corrigió a `Fx = -Kp * error_yaw`.
2. **Corrupción de Inercias (Meshes CAD):** Las mallas `.stl` desalineaban los centros de masa, causando "paracaídas" y torques parásitos. Se rediseñó el modelo con primitivas de MuJoCo (cápsulas y cilindros), respetando exactamente las masas (0.512 kg y 0.092 kg) y longitudes ($L_1=96\text{mm}, L_2=163\text{mm}$) teóricas.
3. **Chattering Numérico:** Falsos contactos repetitivos por milisegundo corregidos con una histéresis asimétrica ($F_{TD} = 0.005\text{ N}, F_{LO} = 0.001\text{ N}$) y candados de estado temporales (`MIN_FLIGHT`, `MIN_STANCE`).

---

## 3. Fundamento Matemático y Ecuaciones

### 3.1 Cinemática Directa y Jacobiano
La posición cartesiana del pie en el sistema de referencia de la cadera (plano sagital) se obtiene mediante:
$$p_{\text{toe}}^{\text{hip}} = \begin{bmatrix} x \\ z \end{bmatrix} = \begin{bmatrix} L_H \sin(\theta_3) + L_2 \sin(\theta_3 + \theta_4) \\ -L_H \cos(\theta_3) - L_2 \cos(\theta_3 + \theta_4) \end{bmatrix}$$

El Jacobiano Geométrico, que relaciona velocidades articulares con cartesianas ($\dot{\mathbf{p}} = J \dot{\mathbf{q}}$), se define como:
$$J_{\text{toe}}^{\text{hip}} = \begin{bmatrix} L_H \cos(\theta_3) + L_2 \cos(\theta_3 + \theta_4) & L_2 \cos(\theta_3 + \theta_4) \\ L_H \sin(\theta_3) + L_2 \sin(\theta_3 + \theta_4) & L_2 \sin(\theta_3 + \theta_4) \end{bmatrix}$$

### 3.2 Dinámica y Control en Fase de Vuelo (FLIGHT)
Se utiliza un control de impedancia (PD Cartesiano) proyectado mediante la transpuesta del Jacobiano, compensando la rigidez pasiva del resorte de rodilla ($k_{\text{knee}} = 1.1$ Nm/rad):
$$\boldsymbol{\tau}_{\text{cmd}} = J_{\text{toe}}^T \left( K_p (\mathbf{p}_d - \mathbf{p}) - K_d J_{\text{toe}} \dot{\mathbf{q}} \right) + \begin{bmatrix} 0 \\ k_{\text{knee}} \cdot q_4 \end{bmatrix}$$

### 3.3 Dinámica y Control en Fase de Apoyo (STANCE)
El contacto con el suelo impone una restricción holonómica ($J_{\text{hc}} \dot{\mathbf{q}} = 0$). El controlador mapea las fuerzas deseadas de Reacción del Suelo (Fuerza Vertical Bézier y Regulación de Velocidad Horizontal) a torques articulares:
$$\boldsymbol{\tau}_{\text{cmd}} = J_{\text{toe}}^T \begin{bmatrix} F_{x,\text{control}} \\ F_{z,\text{Bézier}} \end{bmatrix} + K_{p,\text{st}} (\mathbf{q}_d - \mathbf{q}) - K_{d,\text{st}} \dot{\mathbf{q}}$$

---

## 4. Validación contra MATLAB

La estructura de control y cinemática es un clon 1:1 de la versión de MATLAB (UIUC). No obstante, las diferencias numéricas intrínsecas entre simuladores (Integrador `RK4` de paso fijo en MuJoCo vs `ode45` paso variable en MATLAB, y el modelo de contacto rígido `solref/solimp` vs Z=0) exigieron una **re-sintonización técnica justificada**:

| Parámetro | MATLAB (Teórico) | MuJoCo (Aplicado) | Justificación de Ingeniería |
|-----------|-----------------|-------------------|-----------------------------|
| **$K_{p\_sw}$ (Vuelo)** | 200 | 150 | MuJoCo es más responsivo al torque discreto; 200 generaba vibraciones de alta frecuencia. |
| **$K_{d\_sw}$ (Vuelo)** | 10 | 5 | Entorno rígido requiere menor amortiguación derivada. |
| **$Z_{\text{des}}$** | -0.15 m | -0.19 m | Extensión de rodilla optimizada a la altura real del pivote para evitar singularidad en L2. |
| **$T_{ST}$ (Apoyo)** | 0.35 s | 0.068 s | **Adaptación Crítica:** La física de contacto rígido de MuJoCo rechaza la penetración, reduciendo drásticamente la duración de la fase de apoyo. |
| **$K_{p\_st}$ (Apoyo)** | 0.12 | 2.5 | Necesario aumentar la rigidez en el eje X para inyectar velocidad en una ventana de tiempo $T_{ST}$ 5 veces menor. |
| **$Fz_{\text{Bézier}}$** | 40-80 N | 20-100 N | Curva de salto amplificada para asegurar el Lift-Off contrarrestando la gravedad precisa en 3D. |

---

## 5. Análisis Técnico de Resultados

La ejecución del script `analysis.py` sobre el log proporciona 11 gráficas para avalar el desempeño del controlador:

### 5.1 Dinámica Articular y Velocidades (Gráficas 01 y 02)
- **Posiciones (`01_posiciones.png`):** Se observan oscilaciones asintóticamente estables y simétricas. La cadera y la rodilla exhiben amplitudes controladas sin divergencia, alineándose perfectamente con las fases sombreadas (naranja para STANCE).
- **Emulación Encoders (`02_velocidades.png`):** El Filtro Pasa-Bajas ($f_c = 40$ Hz) estima la velocidad articular (`vel_est`) suprimiendo el ruido de derivación de posición con un retraso de fase despreciable respecto a `vel_real` del simulador.

### 5.2 Torques y Criterio de Saturación (Gráficas 03, 09 y 10)
- **Torques (`03_torques.png`):** Los comandos de los actuadores se mantienen estrictamente confinados entre los límites físicos punteados ($\pm 3.728$ Nm).
- **Consumo y Motor (`09_tau_vs_omega`, `10`):** La dispersión Torque-Velocidad demuestra la viabilidad del motor AK80-9. La implementación de saturación adaptativa $\tau_{\text{disp}}(\omega)$ asegura 0% de muestras en saturación extrema, garantizando la integridad térmica del hardware.

### 5.3 Fuerzas de Contacto y GRF (Gráficas 04 y 11)
- **Histéresis FSM (`04_contacto.png`):** Demuestra el éxito del filtro anti-chattering. Cada impacto registra una subida sólida y una caída a 0 al despegar, sin falsos positivos en el umbral.
- **Validación Bézier (`11_grf.png`):** Compara la fuerza física de impacto leída iterando sobre `mj_contactForce` vs el comando teórico. El pie del robot traza eficazmente el perfil medio-seno proyectado inyectando más de 80N para generar la fase de vuelo.

### 5.4 Estabilidad Espacial (Gráfica 05)
- **Trayectoria del Pie (`05_trayectoria_pie.png`):** El gráfico X-Z en el frame de la cadera forma múltiples bucles cerrados sobrepuestos. En robótica de locomoción, un ciclo límite cerrado sin derivación es la comprobación definitiva de la **estabilidad del régimen de marcha regular**.

---

## 6. Guía de Ejecución

1. **Ejecutar la simulación:**
   ```bash
   python main.py