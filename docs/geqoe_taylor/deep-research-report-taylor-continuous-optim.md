# Implementación de optimización de maniobras de empuje continuo en GEqOE con heyoka

## Resumen ejecutivo

Este informe desarrolla una formulación rigurosa para reemplazar un esquema previo de maniobras impulsivas por **empuje continuo** dentro del marco de los **Generalized Equinoctial Orbital Elements (GEqOE)**, incorporando el empuje como parte del **término no conservativo** \(\mathbf{P}\) en la descomposición de fuerzas \(\mathbf{F}=\mathbf{P}-\nabla U\). La piedra angular es que, en GEqOE, las perturbaciones **derivables de potencial** \(U(\mathbf{r},t)\) se “embeben” en la definición de los elementos, mientras que el empuje continuo entra explícitamente mediante las proyecciones \((P_r,P_f,P_h)\) en el triedro orbital \((\mathbf{e}_r,\mathbf{e}_f,\mathbf{e}_h)\) y afecta tanto a la variación de energía \(\dot{\mathcal{E}}\) como a los términos que contienen \(F_r\) y \(F_h\). fileciteturn0file0

Se presenta una **parametrización general del control** basada en un vector de parámetros \(\mathbf{K}\) (o “K-expansion”), entendida como una expansión en bases (por tramos, polinómica/splines, o series de Fourier en anomalía), que habilita métodos de optimización directa (NLP) apoyados en derivadas. Se conectan explícitamente: (i) la dinámica GEqOE con empuje continuo, (ii) la dinámica de masa y límites de \(T\), potencia e \(I_{sp}\), y (iii) la transcripción numérica (single shooting, multiple shooting, colocation y pseudospectral). citeturn0search3turn0search2turn2search2turn2search0

Para la implementación, se justifica el uso de **heyoka** como integrador de **método de Taylor** con **salida densa** y **detección robusta de eventos** (útil para conmutaciones, cambios de modo, eclipses, o restricciones activas), junto con su **sistema simbólico y diferenciación automática** para construir ecuaciones variacionales (sensibilidades) y derivadas simbólicas. fileciteturn0file1 citeturn1search2turn1search5turn0search0turn4search0turn4search3

Se incluyen tablas comparativas (parametrizaciones, transcripciones y resolutores), estimaciones de coste computacional (orden de magnitud y escalado), estrategias de inicialización (homotopías y “suavizado” de impulsos), regularización para evitar no suavidades, y un conjunto de experimentos/validaciones con métricas esperadas (coherencia con propagación cartesiana, verificación de gradientes, sensibilidad a tolerancias y robustez ante restricciones). fileciteturn0file2 fileciteturn0file3 citeturn3search15turn0search9turn3search26

**Supuestos abiertos** (parámetros a definir por misión): modelo gravitatorio (dos cuerpos vs n-cuerpos, armónicos, terceros cuerpos), magnitud máxima de empuje \(T_{\max}\), ley de eficiencia \(\eta(\cdot)\), límites de potencia \(P_{\max}(t)\), \(I_{sp}(t)\) (constante o dependiente de potencia), masa inicial \(m_0\), ventanas de operación (eclipse), perfil temporal y condiciones de frontera (órbita objetivo, tiempo libre o fijo).

## Formulación matemática con GEqOE y empuje continuo

### Dinámica base y separación conservativo/no conservativo

Se considera la ecuación de movimiento (por unidad de masa) en un marco inercial \(\Sigma=\{O;\mathbf{e}_x,\mathbf{e}_y,\mathbf{e}_z\}\):
\[
\ddot{\mathbf{r}} \;=\; -\frac{\mu}{r^3}\mathbf{r}\;+\;\mathbf{F}(\mathbf{r},\dot{\mathbf{r}},t)
\quad,\qquad
\mathbf{F} \;=\;\mathbf{P}(\mathbf{r},\dot{\mathbf{r}},t)\;-\;\nabla U(\mathbf{r},t),
\]
donde \(U\) agrupa perturbaciones derivables de potencial (p.ej., terceros cuerpos modelados como potencial perturbador, oblaticidad, etc.) y \(\mathbf{P}\) agrupa fuerzas no derivables de potencial (arrastre, SRP modelada fuera de potencial, **empuje continuo**, etc.). Esta separación es la base conceptual de GEqOE. fileciteturn0file0 fileciteturn0file2

Se define el triedro orbital \(\Sigma_{or}=\{O;\mathbf{e}_r,\mathbf{e}_f,\mathbf{e}_h\}\) con
\[
\mathbf{e}_r=\frac{\mathbf{r}}{r},\qquad
\mathbf{e}_h=\frac{\mathbf{r}\times \dot{\mathbf{r}}}{\|\mathbf{r}\times \dot{\mathbf{r}}\|},\qquad
\mathbf{e}_f=\mathbf{e}_h\times \mathbf{e}_r,
\]
y las proyecciones escalares \(P_r=\mathbf{P}\cdot \mathbf{e}_r\), \(P_f=\mathbf{P}\cdot \mathbf{e}_f\), \(P_h=\mathbf{P}\cdot \mathbf{e}_h\), análogamente para \(\mathbf{F}\): \(F_r=\mathbf{F}\cdot \mathbf{e}_r\), \(F_h=\mathbf{F}\cdot \mathbf{e}_h\). fileciteturn0file0 fileciteturn0file2

### Definiciones GEqOE relevantes para empuje continuo

El trabajo que introduce los GEqOE define cantidades generalizadas que incorporan \(U\) en la geometría no osculatriz (elipse “no-osculante”). Entre ellas, son críticas para la implementación:

- Potencial efectivo:
\[
U_{eff}\;=\;\frac{h^2}{2r^2}+U(\mathbf{r},t),
\]
- “momento angular generalizado”:
\[
c=\sqrt{2r^2 U_{eff}},
\]
- relación útil:
\[
c^2=\mu \,\rho,\qquad \rho=a(1-g^2),
\]
donde \(a\) y \(g\) son el semieje mayor y la excentricidad generalizada (magnitud del vector de Laplace generalizado). fileciteturn0file0

El conjunto GEqOE considerado en ese marco es (se usa notación caligráfica para no confundir con la longitud verdadera):
\[
\mathbf{x} \equiv (\nu,\,p_1,\,p_2,\,\mathcal{L},\,q_1,\,q_2),
\]
donde \(\nu\) es una generalización del movimiento medio, \((p_1,p_2)\) parametrizan el vector de excentricidad generalizada en el marco equinoctial, \(\mathcal{L}\) es la longitud media generalizada, y \((q_1,q_2)\) parametrizan la orientación del plano orbital (análogos equinoctiales). fileciteturn0file0

Para evaluar la dinámica, también aparecen:
\[
a=\left(\frac{\mu}{\nu^2}\right)^{1/3},\qquad
\beta=\sqrt{1-p_1^2-p_2^2},\qquad
\alpha=\frac{1}{1+\beta},
\]
y la relación para el momento angular “físico”:
\[
h = \sqrt{c^2 - 2 r^2 U(\mathbf{r},t)}.
\]
fileciteturn0file0

### Ecuaciones no conservativas GEqOE con empuje continuo embebido en \(\mathbf{P}\)

El punto clave para la sustitución “impulsos \(\to\) empuje continuo” es que, en GEqOE, la variación de energía total (por unidad de masa) \(\dot{\mathcal{E}}\) depende explícitamente de \(\mathbf{P}\) (no de \(-\nabla U\), salvo por \(\partial U/\partial t\)):

\[
\dot{\mathcal{E}}=\frac{\partial U}{\partial t} + \dot r \,P_r + \frac{h}{r}P_f.
\]
fileciteturn0file0

Con ello, la dinámica temporal de los elementos (forma compacta) queda:

\[
\dot{\nu} = -3\left(\frac{\nu}{\mu^2}\right)^{1/3}\dot{\mathcal{E}},
\]
fileciteturn0file0

\[
\dot p_1
=
p_2\!\left(\frac{h-c}{r^2}-w_h\right)
+\frac{1}{c}\left(\frac{X}{a}+2p_2\right)\left(2U-rF_r\right)
+\frac{1}{c^2}\Big[Y(r+\rho)+r^2 p_1\Big]\dot{\mathcal{E}},
\]
\[
\dot p_2
=
p_1\!\left(w_h-\frac{h-c}{r^2}\right)
-\frac{1}{c}\left(\frac{Y}{a}+2p_1\right)\left(2U-rF_r\right)
+\frac{1}{c^2}\Big[X(r+\rho)+r^2 p_2\Big]\dot{\mathcal{E}},
\]
fileciteturn0file0

\[
\dot{\mathcal{L}}
=
\nu+\frac{h-c}{r^2}-w_h
+\frac{1}{c}\Big[\frac{1}{\alpha}+\alpha\left(1-\frac{r}{a}\right)\Big]\left(2U-rF_r\right)
+\frac{r\dot r\,\alpha}{\mu c}(r+\rho)\dot{\mathcal{E}},
\]
fileciteturn0file0

\[
\dot q_1=\frac{1}{2}w_Y(1+q_1^2+q_2^2),
\qquad
\dot q_2=\frac{1}{2}w_X(1+q_1^2+q_2^2),
\]
fileciteturn0file0

donde \(X,Y\) son las coordenadas de \(\mathbf{r}\) en el marco equinoctial \(\Sigma_{eq}\) (i.e., \(X=\mathbf{r}\cdot \mathbf{e}_X\), \(Y=\mathbf{r}\cdot \mathbf{e}_Y\)), y las velocidades angulares del marco equinoctial quedan:

\[
w_X=\frac{X}{h}F_h,\qquad
w_Y=\frac{Y}{h}F_h,\qquad
w_h=w_X q_1-w_Y q_2.
\]
fileciteturn0file0

**Cómo entra el empuje continuo**: si la aceleración de empuje se modela como \(\mathbf{a}_T\), entonces \(\mathbf{P}=\mathbf{a}_T + \mathbf{P}_{otros}\). Por tanto, \((P_r,P_f,P_h)\) pasan a depender del control y aparecen:

- directamente en \(\dot{\mathcal{E}}\) vía \(\dot r P_r + \frac{h}{r}P_f\),
- indirectamente en \(F_r\) y \(F_h\), ya que \(\mathbf{F}=\mathbf{P}-\nabla U\) y por tanto \(F_r = P_r - (\nabla U)\cdot \mathbf{e}_r\), \(F_h = P_h - (\nabla U)\cdot \mathbf{e}_h\). fileciteturn0file0

### Modelo de empuje y dinámica de masa (parámetros abiertos)

Un modelo estándar (suficiente para optimización de bajo empuje) es:
\[
\mathbf{a}_T(t)=\frac{T(t)}{m(t)}\mathbf{u}(t),\qquad \|\mathbf{u}(t)\|=1,\qquad 0\le T(t)\le T_{\max}(t),
\]
y el consumo:
\[
\dot m(t)=-\frac{T(t)}{g_0 I_{sp}(t)}.
\]
La interpretación física de \(I_{sp}\) y su relación con empuje y caudal másico es clásica en propulsión. citeturn3search0turn3search19

Para propulsión eléctrica (SEP/EP), si se controla por potencia, una relación ampliamente usada proviene de \(P_{jet}=\tfrac{1}{2}\dot m v_e^2\) y \(T=\dot m v_e\), con eficiencia total \(\eta = P_{jet}/P_{in}\), lo que lleva a:
\[
T = \frac{2\eta\,P_{in}}{v_e}=\frac{2\eta\,P_{in}}{g_0 I_{sp}}.
\]
Este tipo de vínculo “potencia–empuje–\(I_{sp}\)” es central en EP y se expone en textos de referencia de propulsión eléctrica. citeturn3search0turn3search22

**Decisión de modelado** (abierta): mantener \(I_{sp}\) constante, o usar \(I_{sp}(P)\) y \(\eta(P)\) calibrados a un thruster concreto (curvas de rendimiento); esto cambia tanto restricciones de trayectoria como el coste de combustible/tiempo.

## Parametrización del control y variable independiente

### Concepto de “\(\mathbf{K}\)-expansion” para empuje continuo

Sea \(\mathbf{K}\in\mathbb{R}^{n_K}\) un vector de parámetros de control. Una parametrización general para el empuje en el marco orbital puede escribirse como:
\[
\begin{aligned}
P_r(\tau) &= \sum_{i=1}^{N} K_{r,i}\,\phi_i(\tau),\\
P_f(\tau) &= \sum_{i=1}^{N} K_{f,i}\,\phi_i(\tau),\\
P_h(\tau) &= \sum_{i=1}^{N} K_{h,i}\,\phi_i(\tau),
\end{aligned}
\qquad \tau\in[0,1],
\]
donde \(\tau\) es un parámetro normalizado del “tiempo” del arco (o de una anomalía), y \(\{\phi_i\}\) son funciones base (tramos, polinomios, splines, Fourier, etc.). Esta forma conecta directamente con GEqOE, porque \((P_r,P_f,P_h)\) son exactamente las entradas requeridas en \(\dot{\mathcal{E}}\), \(F_r\), \(F_h\). fileciteturn0file0

Alternativamente, puede parametrizarse **magnitud + dirección**:
\[
T(\tau)=T_{\max}(\tau)\,\sigma(\tau), \quad \sigma\in[0,1],\qquad
\mathbf{u}(\tau)=\mathbf{u}(\gamma(\tau),\delta(\tau)),
\]
con \(\gamma,\delta\) ángulos (por ejemplo en el triedro orbital), evitando imponer \(\|\mathbf{u}\|=1\) de forma dura. Esta opción suele requerir regularización para evitar no suavidades cuando \(\sigma\to 0\).

### Elección de variable independiente: \(t\), \(\tau\), anomalías, y el papel de \(\mathcal{K}\)

En GEqOE aparece una anomalía auxiliar (en el artículo se usa \(K\), aquí la denoto \(\mathcal{K}\) para distinguir de \(\mathbf{K}\) de control) ligada por una ecuación tipo Kepler:
\[
\mathcal{L} = \mathcal{K} + p_1\cos\mathcal{K} - p_2\sin\mathcal{K}.
\]
fileciteturn0file0

**Implicación práctica**: si se propaga en tiempo con estado \((\nu,p_1,p_2,\mathcal{L},q_1,q_2)\), típicamente hay que evaluar \(\sin\mathcal{K}\), \(\cos\mathcal{K}\) y variables geométricas derivadas, y por tanto resolver (o actualizar) \(\mathcal{K}\) en cada evaluación. El artículo describe procedimientos para pasar de elementos a posición/velocidad y sugiere sustituciones para evaluar términos de forma eficiente. fileciteturn0file0

Hay tres estrategias (todas compatibles con heyoka):

1. **Resolver \(\mathcal{K}\) por iteración (Newton) dentro del RHS**.
   Pros: implementación directa. Contras: complica derivadas (hay que diferenciar a través del solver), y puede introducir ruido numérico si no se hace con cuidado.

2. **Aumentar el estado con \(\mathcal{K}\)** y propagar una ODE para \(\dot{\mathcal{K}}\) derivada por diferenciación de la ecuación de Kepler generalizada. Esto elimina búsquedas iterativas por paso y preserva diferenciabilidad “nativa” del RHS (solo hay \(\sin,\cos\)). Es especialmente atractivo si se quiere AD/sensibilidades consistentes.

3. **Cambiar de elemento rápido**: usar variantes con longitud verdadera como elemento rápido (p.ej., M-GEqOE), lo que puede mejorar robustez en ciertos regímenes y reducir complejidad de evaluación. El trabajo de M-GEqOE emplea explícitamente \(L\) (longitud verdadera) como variable rápida y da ecuaciones en forma apta para perturbaciones y fuerzas externas. fileciteturn0file2

### Ejemplo de parametrización en serie: Fourier en anomalía (alineada con “K-expansion”)

Un ejemplo relevante (aplicable como “K-expansion” de control) es expresar el empuje (o aceleración de empuje) como series truncadas de Fourier en anomalía excéntrica \(E\):
\[
u_R(E)\approx \sum \kappa_{R,j}\cos(jLE)+\sum \kappa_{R,j'}\sin(jLE),
\]
y análogamente \(u_S(E)\), \(u_W(E)\), con coeficientes \(\kappa\) como variables de optimización. Esta idea se usa para aproximar perfiles de empuje y derivar expresiones analíticas/semianalíticas en elementos medios. fileciteturn0file3

Aunque ese trabajo se centra en detección/caracterización (no en transferencia óptima), la estructura es muy útil para: (i) imponer suavidad, (ii) controlar ancho de banda del control, y (iii) reducir dimensionalidad.

### Tabla comparativa de parametrizaciones de control

| Parametrización (\(\mathbf{K}\)) | Variable típica | Ventajas | Riesgos / desventajas | Idoneidad con GEqOE + heyoka |
|---|---|---|---|---|
| Tramos constantes (ZOH) en \(\mathbf{a}_T\) o \((\sigma,\gamma,\delta)\) | tiempo normalizado \(\tau\) | Implementación simple; controla discontinuidades en nodos; buena para multiple shooting | Control no suave \(\Rightarrow\) reducción de paso; puede inducir “bang-bang” numérico; requiere muchos tramos para precisión | Muy alta: segmentar integra bien con eventos y arcos |
| Polinomios (Lagrange/Chebyshev) por arco | \(\tau\in[0,1]\) | Control suave; pocos parámetros; compatible con colocation/pseudospectral | Posible oscilación (Runge) si global; escalado delicado | Alta, especialmente con transcripción simultánea |
| B-splines / splines cúbicos | \(\tau\) | Suavidad local; control robusto; regularización natural | Elección de nudos; restricciones \(\sigma\in[0,1]\) requieren mapeos suaves | Muy alta (buen compromiso robustez/dimens.) |
| Fourier en anomalía (coef. \(\kappa\)) | \(E\) o \(\mathcal{K}\) | Coeficientes interpretables; control banda limitada; útil en órbitas multi-vuelta | Menos apto si hay eclipses/ventanas no periódicas; difícil con restricciones locales | Alta si la misión admite periodicidad o arcos por revolución fileciteturn0file3 |
| Control continuo libre en nodos de colocation (valores \(u_i\)) | nodos de colocation | Máxima flexibilidad; fácil imponer límites en nodos | Dimensión grande; riesgo de sobreajuste; necesita buena malla | Alta en problemas duros con restricciones activas citeturn2search2turn2search0 |

## Planteamiento del problema de control óptimo y transcripción

### OCP continuo en GEqOE con masa y control \(\mathbf{K}\)

Defínase el estado ampliado:
\[
\mathbf{z}(t)=\big(\nu,p_1,p_2,\mathcal{L},q_1,q_2,m\big),
\]
con dinámica:
\[
\dot{\mathbf{z}} = \mathbf{f}\!\left(t,\mathbf{z},\mathbf{u}(t)\right),
\]
donde \(\mathbf{f}\) se construye combinando las ecuaciones GEqOE anteriores y \(\dot m\). fileciteturn0file0

El control puede ser \(\mathbf{u}(t)\) (dirección + throttle), o indirectamente \(\mathbf{u}(t;\mathbf{K})\) vía expansión en bases.

**Costes típicos** (selección explícita por misión):

- **Mínimo propelente** (máximo \(m(t_f)\)):
\[
\min \; J = m(t_0)-m(t_f)
\quad \text{o}\quad
\min \; J = \int_{t_0}^{t_f} \frac{T(t)}{g_0 I_{sp}(t)}\,dt.
\]
- **Mínimo tiempo** (tiempo libre):
\[
\min \; J = t_f-t_0,
\]
con límites \(T\le T_{\max}\), potencia, etc.
- **Combinado** (ponderado / multiobjetivo escalarizado):
\[
\min\; J = w_t(t_f-t_0) + w_m(m(t_0)-m(t_f)) + w_s\int \|\dot{\mathbf{u}}\|^2 dt,
\]
donde el último término regulariza suavidad (evita controles “dentados”). La literatura de optimización de bajo empuje suele recomendar este tipo de regularizaciones cuando se busca convergencia robusta en métodos directos. citeturn0search9turn3search26turn0search3

**Restricciones**:

- Condiciones de contorno:
\[
\mathbf{z}(t_0)=\mathbf{z}_0,\qquad
\mathbf{c}\big(\mathbf{z}(t_f)\big)=\mathbf{0}
\]
(p.ej., igualar un conjunto de elementos objetivo, o posición/velocidad).
- Límites de control:
\[
0\le \sigma(t)\le 1,\quad 0\le T(t)\le T_{\max}(t),\quad P_{in}(t)\le P_{\max}(t),\quad I_{sp}(t)\in[I_{\min},I_{\max}].
\]
- Restricciones de trayectoria (desigualdades):
\[
g(\mathbf{z}(t),\mathbf{u}(t)) \le 0
\]
(p.ej., altura mínima, no entrar en sombra si el thruster requiere potencia solar, límites térmicos, apuntado). Restricciones de este tipo aparecen con frecuencia en problemas SEP y en extensiones del problema de Edelbaum con desigualdades. citeturn3search9turn3search28

### Directo vs indirecto: implicaciones para “\(\mathbf{K}\)-expansion”

- **Métodos indirectos** (PMP, BVP): precisos pero requieren inicializar coestados y gestionar conmutaciones; esto suele ser el cuello de botella práctico en bajo empuje. citeturn0search3turn0search25
- **Métodos directos**: transcriben a un NLP; son el estándar práctico moderno por robustez y facilidad para desigualdades y restricciones multipunto. citeturn0search3turn0search29turn2search2

Una “\(\mathbf{K}\)-expansion” se alinea naturalmente con métodos directos (porque \(\mathbf{K}\) pasa a ser el vector de decisión), mientras que en métodos indirectos \(\mathbf{K}\) sería más bien una reducción dimensional del control para facilitar el BVP.

### Esquemas de transcripción y comparación

| Esquema | Idea | Pros | Contras | Escalado aproximado (decisión/estructura) |
|---|---|---|---|---|
| Single shooting | NLP solo en \(\mathbf{K}\); la dinámica se integra “dentro” | Pocas variables; fácil de implementar | Muy sensible; inestable si la dinámica amplifica errores; difíciles desigualdades internas | Variables \(\sim n_K\); coste eval \(\sim N_{int}\) integraciones |
| Multiple shooting | \(\mathbf{K}\) + estados en nodos; continuidad como restricción | Mucho más robusto; gestiona discontinuidades y arcos; apto para restricciones globales | NLP mayor; requiere jacobianos dispersos | Variables \(\sim n_K + N_{nodos}n_x\) citeturn0search2turn0search21 |
| Colocation (Hermite–Simpson / polinomios cúbicos) | Aproxima estado por polinomios; impone dinámica en puntos | Muy usado en aeroespacio; desigualdades por nodo; jacobianos dispersos | Ajuste de malla; puede requerir refinamiento | NLP grande pero disperso citeturn2search2turn2search6 |
| Pseudospectral (LGR/LG/LGL, hp) | Colocación global/por elementos usando cuadraturas | Alta precisión con pocos nodos; buena para OCP suaves | Menos natural con no suavidades o eventos frecuentes; implement. más compleja | NLP grande; buenas propiedades espectrales citeturn2search0turn2search1turn2search13 |

### Solvers de NLP y manejo de desigualdades

Para el NLP resultante, un enfoque común es usar un método de **punto interior** tipo Ipopt, diseñado para problemas grandes con restricciones y estructura dispersa. citeturn2search3turn2search11
La elección de solver depende de (i) tamaño, (ii) necesidad de Hessiano exacto o aproximado, y (iii) robustez ante mal condicionamiento; por ejemplo, Ipopt introduce un método de punto interior con filtro line-search y múltiples opciones de escalado/tolerancias. citeturn2search3turn2search18turn2search15

## Implementación con heyoka: integración Taylor, AD, sensibilidades y eventos

### Capacidades relevantes de heyoka para esta arquitectura

El integrador basado en método de Taylor implementado en heyoka destaca por:

- cálculo eficiente de coeficientes de Taylor mediante diferenciación automática y compilación JIT, con salida densa “de coste cero” (muy útil para evaluación de integrales de coste, muestreo fino sin reintegrar, y detección de eventos), fileciteturn0file1
- soporte explícito de eventos con técnicas de búsqueda de raíces sobre la serie de Taylor, lo que permite localizar con precisión el instante de activación de una condición (p.ej., cruces geométricos, entrada/salida de sombra, cambios de modo). citeturn0search0turn0search4turn0search31
- un sistema simbólico con derivación simbólica (`diff`) y cálculo de tensores de derivadas (`diff_tensors`) en la API Python. citeturn1search1
- soporte “built-in” de ecuaciones variacionales (sensibilidades) con generación de un sistema variacional (`var_ode_sys`) y recomendaciones prácticas como `compact_mode` cuando crece el tamaño simbólico. citeturn1search5turn1search8turn4search3
- paralelización y vectorización: batch mode (SIMD) y ensemble/parallel mode para muchas integraciones independientes (muy útil en multiarranque, exploración de inicializaciones, o evaluación paralela de arcos). citeturn4search0turn4search7turn4search10

### Construcción de la dinámica GEqOE + empuje en el sistema simbólico

En términos de ingeniería de implementación, se recomienda separar:

1. **Módulo geométrico**: a partir de \((\nu,p_1,p_2,\mathcal{L},q_1,q_2)\) y \(t\), computar \(a,c,\rho,\alpha,\beta\), y reconstruir las cantidades necesarias para \(r,\dot r,X,Y\) (y, si procede, \(\mathcal{K}\)). El artículo describe procedimientos “elementos \(\to\) posición/velocidad” y sustituciones para evaluar términos que aparecen en \(\dot{\mathcal{L}}\). fileciteturn0file0

2. **Módulo de perturbaciones conservativas**: evaluar \(U(\mathbf{r},t)\), \(\nabla U(\mathbf{r},t)\), y si aplica \(\partial U/\partial t\). El trabajo M-GEqOE muestra el uso de potenciales perturbadores de terceros cuerpos y su expansión en polinomios de Legendre, y discute cuándo ciertas contribuciones se modelan como potencial o como fuerza externa por razones numéricas. fileciteturn0file2

3. **Módulo de control**: mapear \(\mathbf{K}\mapsto (P_r,P_f,P_h)\) (o a \(\mathbf{a}_T\) en un marco dado y luego proyectar). Esto es donde la “K-expansion” entra estrictamente.

4. **Módulo GEqOE RHS**: implementar literalmente (45)–(51) con \(\dot{\mathcal{E}}\), \(F_r\), \(F_h\), \(w_X,w_Y,w_h\). fileciteturn0file0

### Derivadas para optimización: sensibilidades por ecuaciones variacionales

Para optimización basada en gradiente, el objeto clave es el Jacobiano del “endpoint map”:
\[
\mathbf{G}(\mathbf{K}) =
\begin{bmatrix}
\mathbf{c}(\mathbf{z}(t_f;\mathbf{K}))\\
J(\mathbf{K})
\end{bmatrix},
\qquad
\nabla_{\mathbf{K}}\mathbf{G}.
\]

En aproximación **forward (variacional)**, si \(\mathbf{z}=\mathbf{z}(t;\mathbf{K})\) y \(\dot{\mathbf{z}}=\mathbf{f}(t,\mathbf{z},\mathbf{K})\), las sensibilidades \(\mathbf{S}=\partial \mathbf{z}/\partial \mathbf{K}\) satisfacen:
\[
\dot{\mathbf{S}} = \frac{\partial \mathbf{f}}{\partial \mathbf{z}}\mathbf{S} + \frac{\partial \mathbf{f}}{\partial \mathbf{K}}.
\]
heyoka permite construir estas ecuaciones variacionales: bien manualmente con derivadas simbólicas de \(\mathbf{f}\) (ejemplo típico de \(\partial f_i/\partial x_j\)), o automáticamente usando su infraestructura variacional. citeturn1search0turn1search2turn1search5

**Recomendación práctica**:
- Si \(n_K\) es moderado (p.ej. decenas–centenas), forward variacional suele ser viable.
- Si \(n_K\) crece mucho (miles), puede convenir un enfoque adjunto (reverse) o una transcripción simultánea (colocation) que explota dispersidad del NLP (según la guía general de OCP con NLP disperso). citeturn0search3turn2search2

### Eventos y conmutaciones: utilidad directa en restricciones activas

La detección de eventos es útil en al menos tres lugares:

- **cambios de modo**: encendido/apagado del thruster (si se modela como conmutación), cambios de límites \(T_{\max}(t)\), límites térmicos.
- **eclipses**: si \(P_{\max}(t)\) depende de iluminación, los eventos de entrada/salida de sombra se convierten en discontinuidades o restricciones activas.
- **restricciones geométricas**: cruces de plano, paso por periapsis, etc.

heyoka documenta un sistema de eventos que usa la serie de Taylor y búsqueda de raíces polinómicas para localizar el evento dentro del paso y con sobrecoste moderado. citeturn0search0turn0search16turn0search31

### Pseudocódigo de arquitectura (multiple shooting recomendado)

A continuación se da un esquema genérico (sin ligarlo a una API concreta) coherente con el flujo “\(\mathbf{K}\)-NLP + integración”:

```text
Inputs:
  - z0: estado inicial (GEqOE + masa)
  - target: condiciones finales (igualdad y/o desigualdad)
  - model_U: potencial U(r,t) + ∇U + ∂U/∂t (si aplica)
  - control_basis: {phi_i} y mapeo K -> P_r,P_f,P_h (o -> a_T)
  - mesh: partición en arcos [t0,t1],...,[t_{M-1},t_M]
  - decision vector: y = [K, z_nodes(1..M-1), tf, ...] (según formulación)

Function EvaluateNLP(y):
  1) Decode y -> K, node states, tf, ...
  2) For each arc j in 0..M-1:
       - set initial state z_j(0):
            if j=0: z0
            else:   z_node[j]
       - integrate GEqOE dynamics with heyoka from t_j to t_{j+1}
            dynamics uses:
              - reconstruct r, frames, X,Y,...
              - compute U, ∇U, ∂U/∂t
              - compute thrust projections from K at time t
              - assemble dot{z}
            optionally integrate variational equations for sensitivities
       - obtain final state z_j(1)
       - add continuity constraint: z_j(1) - z_node[j+1] = 0 (except last)
  3) Terminal constraint: c(z_M(1)) = 0 (and inequalities)
  4) Objective J:
       - if fuel: use m(tf) or integral from dense output
       - if time: tf
       - regularize control if needed
  5) Return:
       - objective value J
       - constraints vector g
       - Jacobian/Hessian via sensitivities and/or AD

Solve NLP with (e.g.) interior-point method:
  y* = argmin J(y) s.t. g_L <= g(y) <= g_U, y_L <= y <= y_U

Postprocess:
  - reconstruct continuous thrust profile
  - validate by Cartesian propagation
  - compute metrics and sensitivities
```

La idoneidad del enfoque multiple shooting para OCP y su robustez frente a no linealidades está ampliamente documentada, y encaja especialmente bien cuando se quieren “puntos de reseteo” alineados con discontinuidades del control. citeturn0search2turn0search29turn0search25

### Diagrama de flujo (Mermaid) del pipeline propuesto

```mermaid
flowchart TD
  A[Definir modelo dinámico: GEqOE + U + empuje] --> B[Elegir parametrización del control K-expansion]
  B --> C[Elegir transcripción: multiple shooting / colocation / PS]
  C --> D[Construir sistema simbólico en heyoka]
  D --> E[Activar: variacionales (sensibilidades) + compact_mode si procede]
  E --> F[Resolver NLP: restricciones + coste]
  F --> G[Validar: propagación cartesiana + chequeo de gradientes]
  G --> H[Refinar: malla, regularización, homotopía, tolerancias]
  H --> F
```

## Estabilidad, inicialización, restricciones, regularización y rendimiento

### Estabilidad numérica y control del error con método de Taylor

Los métodos de Taylor son explícitos y, como clase, pueden sufrir en problemas fuertemente rígidos; hay resultados que indican estabilidad asintótica al aumentar el orden, y en la práctica órdenes altos permiten abordar problemas moderadamente rígidos. fileciteturn0file1
La estrategia de heyoka (orden y paso adaptativos, salida densa) está pensada para problemas exigentes de mecánica celeste/astrodinámica, incluyendo fuerzas no conservativas. fileciteturn0file1

**Implicación para empuje continuo**: el empuje en sí no suele inducir rigidez, pero sí pueden hacerlo:
- discontinuidades (encendido/apagado, saturaciones),
- perfiles de empuje de alta frecuencia (series con muchos armónicos),
- modelos con ephemerides y potenciales complejos con escalas temporales dispares (cislunar/n-cuerpos). fileciteturn0file2

Cuando haya no suavidades inevitables, segmentar el tiempo (multiple shooting) y alinear nodos con discontinuidades reduce la presión sobre el integrador y mejora la convergencia del NLP. citeturn0search2turn0search3

### Estrategias de inicialización (de impulsivo a continuo)

Para pasar de un método impulsivo previo a empuje continuo, suelen funcionar bien estas rutas:

- **Suavizado de impulsos**: reemplazar cada \(\Delta \mathbf{v}\) por un arco de duración \(\Delta t\) con empuje constante \(T\) y dirección fija, tal que \(\int \mathbf{a}_T dt \approx \Delta \mathbf{v}\). Este “puente” permite iniciar la optimización continua cerca de la solución impulsiva.
- **Homotopía en magnitud de empuje**: resolver primero con empuje pequeño o restricciones relajadas y aumentar gradualmente \(T_{\max}\) o endurecer límites.
- **Homotopía en dimensión del control**: empezar con pocos coeficientes \(\mathbf{K}\) (control “liso”) y aumentar el orden de la expansión o el número de nodos/malla.
Estas estrategias son coherentes con revisiones de enfoques numéricos en bajo empuje y con tesis enfocadas a propagación/óptimo en trayectorias de bajo empuje. citeturn0search9turn3search26

### Regularización y suavizado para evitar “patologías” del NLP

Recomendaciones prácticas (especialmente con límites \(\sigma\in[0,1]\)):

- Penalizar \(L_2\) de control o variación del control:
  \(\int \|\mathbf{u}\|^2 dt\) o \(\int \|\dot{\mathbf{u}}\|^2 dt\), para evitar soluciones altamente oscilatorias que pueden ser artefactos de discretización.
- Usar mapeos suaves para límites:
  \(\sigma=\text{sigmoid}(s)\) o \(\sigma=\frac{1}{2}(1+\tanh s)\) si se prefieren variables no acotadas; o mantener \(\sigma\) acotada y usar punto interior (Ipopt) para respetar límites.
- Si hay eventos de eclipse o potencia, preferir modelado por arcos con disponibilidad constante por arco (o evento explícito), en lugar de una función “dura” no diferenciable. La detección de eventos de heyoka facilita esta partición. citeturn2search3turn0search0turn0search31

### Manejo de desigualdades de trayectoria

- En colocation/pseudospectral: imponer desigualdades en nodos y, si hace falta, usar refinamiento de malla (hp) o chequeo a posteriori de máximos entre nodos (apoyado en salida densa).
- En shooting: desigualdades se tratan mediante penalización/barrier, o introduciendo variables de holgura y restricciones adicionales; los solvers de punto interior están diseñados para \(g_L\le g(x)\le g_U\). citeturn2search3turn2search11turn2search15

### Rendimiento y escalabilidad: estimaciones de coste

En problemas de bajo empuje, el coste total suele venir de:
1) número de integraciones (evaluaciones del NLP),
2) coste por integración (tolerancia, complejidad del RHS, orden Taylor),
3) coste de derivadas (sensibilidades).

heyoka introduce consideraciones específicas:
- **JIT y tamaño simbólico**: `compact_mode` reduce drásticamente tiempo de compilación y memoria, con degradación de rendimiento típicamente \(\lesssim 2\times\) en ciertos ejemplos. citeturn4search3turn4search5
- **Batch mode (SIMD)**: permite integrar varios casos con el mismo sistema (distintas IC/parámetros) casi al coste de uno, aumentando throughput (útil en multiarranque o en evaluación “en paralelo” de varios candidatos \(\mathbf{K}\)). citeturn4search0turn4search2
- **Ensemble/parallel**: útil cuando se necesitan muchas trayectorias independientes; la documentación advierte que la escalabilidad en paralelo puede estar limitada por “memory wall” en sistemas grandes. citeturn4search26turn4search7

#### Tabla de coste esperado (orden de magnitud)

| Elección | Variables decisión | Evaluación por iteración | Derivadas | Coste esperado relativo |
|---|---:|---|---|---|
| Single shooting + \(\mathbf{K}\) pequeño | \(n_K \sim 10^1-10^2\) | 1 integración larga | variacional forward \(O(n_x n_K)\) | Bajo–medio (pero frágil) |
| Multiple shooting (M arcos) | \(n_K + (M-1)n_x\) | \(M\) integraciones cortas | variacional por arco + jacobiano disperso | Medio (robusto) citeturn0search2 |
| Colocation (N nodos) | \(\sim N(n_x+n_u)\) | sin integración interna (residuos) | jacobiano disperso grande | Medio–alto (muy flexible) citeturn2search2turn2search6 |
| Pseudospectral hp | similar a colocation | residuos + cuadraturas | jacobiano/hessiano estructurado | Medio–alto (muy preciso) citeturn2search0turn2search13 |

**Regla práctica**:
- Si el perfil de empuje es relativamente suave y la dimensión del control no muy grande, multiple shooting + variacionales en heyoka suele ser el “punto dulce” (robustez + derivadas exactas).
- Si hay muchas restricciones activas de trayectoria o gran complejidad, colocation/pseudospectral tiende a ganar por estructura dispersa del NLP y mejor tratamiento de desigualdades.

## Experimentos y validación: casos, métricas y resultados esperados

### Principios de validación

Se recomienda validar en tres niveles:

1. **Dinámica**: GEqOE + empuje vs propagación cartesiana con el mismo \(\mathbf{a}_T\).
2. **Optimización**: reproducibilidad y consistencia del óptimo frente a cambios de discretización/tolerancias.
3. **Derivadas**: chequeo sistemático de gradientes (variacionales vs diferencias finitas / complejo-step cuando aplique).

El uso de coordenadas generalizadas (GEqOE/M-GEqOE) para mejorar robustez de propagación bajo perturbaciones y para aplicaciones exigentes (p.ej., cislunar) se ha evaluado comparando contra n-cuerpos cartesiano en trabajos recientes, lo que sirve de guía metodológica para la validación. fileciteturn0file2

### Casos de prueba sugeridos

**Caso A: Dos cuerpos + empuje tangencial constante (sanity check)**
- Modelo: \(U=0\), \(\mathbf{P}=\mathbf{a}_T\) con \(P_f=\text{const}\), \(P_r=P_h=0\), masa constante (o variable).
- Objetivo: elevar \(a\) en tiempo fijo.
- Métricas: error final en elementos vs cartesiano, conservación/deriva esperada, sensibilidad \(\partial a_f / \partial P_f\).

**Caso B: Transferencia tipo Edelbaum (LEO→GEO, cambio de inclinación)**
- Modelo: promedio o modelo completo; empuje continuo.
- Comparar contra aproximaciones conocidas y extensiones modernas (incluyendo restricciones intermedias). La literatura discute Edelbaum y reformulaciones/extensiones (incluyendo restricciones) como bancos de prueba. citeturn3search15turn3search28turn3search9
- Métricas: \(\Delta v\) equivalente, tiempo, consumo, número de revoluciones, convergencia con refinamiento.

**Caso C: Cislunar de alta fidelidad (terceros cuerpos) + empuje**
- Usar el marco M-GEqOE o GEqOE con potenciales de terceros cuerpos; comparar con integración n-cuerpos cartesiana. El trabajo de M-GEqOE da la estructura para incorporar potencial de terceros cuerpos y fuerzas externas, y usa efemérides para posiciones planetarias. fileciteturn0file2
- Métricas: error final, estabilidad numérica, número de pasos, robustez al escalado.

**Caso D: Multi-arco mínimo tiempo Tierra–Luna con bajo empuje**
- Inspirado en formulaciones multi-arco modernas de bajo empuje (mínimo tiempo) donde multiple shooting es natural. citeturn0search1turn0search32
- Métricas: tiempo óptimo, saturación de empuje/potencia, sensibilidad a restricciones.

**Caso E: Control en serie (Fourier) por revolución**
- Parametrización: coeficientes \(\kappa\) en Fourier vs anomalía (por revolución), como ejemplo de “K-expansion”. fileciteturn0file3
- Métricas: número de coeficientes necesarios para alcanzar tolerancia final, suavidad del control, coste computacional.

### Métricas comunes y umbrales recomendados

- **Exactitud dinámica**: \(\|\mathbf{r}_{GEqOE}-\mathbf{r}_{cart}\|\), \(\|\mathbf{v}_{GEqOE}-\mathbf{v}_{cart}\|\) en rejilla (usar salida densa).
- **Exactitud de contorno**: norma de residuo final \(\|\mathbf{c}(\mathbf{z}(t_f))\|\).
- **Calidad del gradiente**: test de dirección aleatoria \(d\):
  \[
  \frac{J(\mathbf{K}+\epsilon d)-J(\mathbf{K})}{\epsilon} \approx \nabla J(\mathbf{K})^\top d
  \]
  para varios \(\epsilon\) en rango (p.ej. \(10^{-6}\)–\(10^{-8}\) en doble).
- **Robustez numérica**: sensibilidad de la solución al refinamiento de malla/orden de control, y al cambio de tolerancia del integrador.
- **Rendimiento**: tiempo por evaluación del NLP, tiempo por iteración, número de evaluaciones hasta convergencia; compilar aparte el coste JIT (amortizable si se reutiliza integrador).

### Resultados esperados (cualitativos)

- GEqOE debería mejorar la propagación frente a elementos equinoctiales clásicos en presencia de perturbaciones derivables de potencial embebidas, según pruebas comparativas reportadas en la literatura que introduce el formalismo. fileciteturn0file0
- En entornos cislunares de alta fidelidad, representaciones generalizadas (M-GEqOE) han mostrado buena concordancia con soluciones n-body cartesianas, lo que sugiere que añadir empuje como \(\mathbf{P}\) mantiene una arquitectura de modelado consistente. fileciteturn0file2
- Parametrizaciones suaves (splines/Fourier) tenderán a reducir pasos de integración y a estabilizar el NLP frente a parametrizaciones con discontinuidades fuertes, especialmente si se exigen restricciones de potencia/eclipses.

## Referencias clave para consulta

Se listan referencias “núcleo” (prioridad a fuentes primarias, documentos adjuntos y material en español cuando está disponible):

- Artículo GEqOE (definiciones y ecuaciones \(\dot\nu,\dot p_1,\dot p_2,\dot{\mathcal{L}},\dot q_1,\dot q_2\); separación \(\mathbf{F}=\mathbf{P}-\nabla U\)). fileciteturn0file0
  Autores: entity["people","Giulio Baù","orbit dynamics researcher"] et al.

- heyoka y método de Taylor (integración de alta precisión, salida densa, AD/JIT). fileciteturn0file1
  Autores: entity["people","Francesco Biscani","astrodynamics researcher"] y entity["people","Dario Izzo","esa act researcher"].

- Documentación heyoka: eventos, batch/parallel, derivadas simbólicas y ecuaciones variacionales. citeturn0search0turn4search0turn4search3turn1search5turn1search1

- M-GEqOE y propagación cislunar de alta fidelidad (forma de ecuaciones con fuerzas no conservativas; potenciales de terceros cuerpos). fileciteturn0file2

- Parametrización de empuje por series (Fourier en anomalía) y coeficientes de empuje como variables (alineable con “K-expansion”). fileciteturn0file3

- OCP por métodos directos y NLP disperso: entity["people","John T. Betts","optimal control author"], SIAM. citeturn0search3 entity["organization","SIAM","professional society, philadelphia"]

- Multiple shooting directo: entity["people","Hans Georg Bock","optimal control researcher"] y Plitt. citeturn0search2turn0search21

- Colocation cúbica en optimización de trayectorias: entity["people","C. R. Hargraves","trajectory optimization author"] y entity["people","Stephen W. Paris","trajectory optimization author"]. citeturn2search2turn2search6

- Pseudospectral (LGR/LG): entity["people","Fariba Fahroo","optimal control researcher"] y entity["people","I. Michael Ross","pseudospectral methods"]. citeturn2search0turn2search8

- Propulsión eléctrica (relación potencia–empuje–\(I_{sp}\)): texto JPL/Descanso. citeturn3search0 entity["organization","NASA Jet Propulsion Laboratory","pasadena, CA, US"]

- Fuentes en español:
  - Encuesta sobre optimización de trayectorias de bajo empuje (repositorio académico español). citeturn0search9
  - Tesis en español sobre propagación y control óptimo de trayectorias espaciales de bajo empuje (UPM). citeturn3search26 entity["organization","Universidad Politécnica de Madrid","madrid, spain"]