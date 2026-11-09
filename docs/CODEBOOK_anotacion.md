# Codebook de anotación — validación de los clasificadores

Materiales de replicación del manuscrito sobre confianza declarada y polaridad
del discurso digital (iCARE / Twitter-X, Colombia).

## Procedencia de las etiquetas

Las métricas de validación que reporta el manuscrito (precisión por clase,
kappa de Cohen y estándar de referencia) provienen de **anotación humana**:

* Dos anotadores humanos etiquetaron de forma **independiente** las tres
  muestras completas — 399 casos de polaridad, 397 de entidad y 400 de origen
  geográfico.
* La anotación fue **a ciegas**: los archivos entregados no contenían la
  predicción del modelo, su confianza ni el número de evidencias geográficas, y
  las filas iban barajadas para que la posición tampoco delatara la clase
  predicha.
* Los 75 casos en que los dos anotadores discreparon (18 de polaridad, 13 de
  origen y 44 de entidad) se **adjudicaron** entre ambos. El estándar de
  referencia es por tanto completo: no se excluyó ningún caso.
* Etiquetas y adjudicaciones estan en `anotados/`. Todas las métricas se
  calculan con `src/s09_evaluar_anotacion.py`, que además exporta
  `s09_calculo_auditable.xlsx`, un libro donde el kappa está en fórmulas de
  Excel para poder rehacer la cuenta sin Python.

**Divulgación.** Las reglas de desempate que siguen se redactaron con ayuda de
un modelo de lenguaje durante una pasada preliminar sobre estas mismas muestras,
antes de la anotación humana. Esa pasada preliminar **no** forma parte de las
métricas reportadas y sus etiquetas no intervienen en ningún resultado del
manuscrito; se conserva solo como diagnóstico interno. Lo que el manuscrito
reporta es exclusivamente la anotación humana descrita arriba.

## Reglas de anotación

Fijadas antes de anotar y entregadas a los dos anotadores.

## Tarea 1 — Polaridad (tono global del tuit)
- Negative: crítica, enfado, miedo, queja, rechazo, denuncia, alarma, burla hostil.
- Neutral: informativo/descriptivo, cifras, avisos, preguntas sin carga, RT de noticia.
- Positive: apoyo, agradecimiento, esperanza, elogio, celebración, ánimo.
- descartar: ilegible, vacío tras limpieza, u otro idioma.
Reglas de desempate:
  R1. La ironía y el sarcasmo se codifican por la INTENCIÓN, no por la literalidad.
  R2. Noticia de cifras malas (muertes, contagios) sin lenguaje evaluativo = Neutral.
      Solo pasa a Negative si el autor añade juicio, alarma o reproche.
  R3. Mezcla de crítica y esperanza: gana la carga dominante; si están
      equilibradas, Neutral.
  R4. Súplicas y peticiones de ayuda = Negative (queja/necesidad).
  R5. Humor no hostil = Positive solo si celebra; si es chiste amargo = Negative.

## Tarea 2 — Entidad (qué ES la cuenta, no de qué habla)
Claves: lider_nacional, lider_regional, autoridad_salud, profesional_salud, oms,
ciencia, famosos, lider_comunitario, allegados, medios, sin_categoria.
Reglas de desempate:
  R1. Ciudadano que opina de política -> sin_categoria, nunca lider_*.
  R2. Cargo público de salud (minsalud, secretaría, INS) -> autoridad_salud,
      aunque la persona sea médica.
  R3. Médico/enfermero que ejerce -> profesional_salud; investigador con
      afiliación académica -> ciencia. Si es ambos, prima el rol declarado primero.
  R4. Periodista individual -> medios. Medio institucional -> medios.
  R5. Alcaldes, gobernadores, concejales, diputados -> lider_regional.
      Presidente, ministros, senadores, cámara -> lider_nacional.
  R6. Cuenta personal anónima sin señales -> sin_categoria.
  R7. Bio vacía o solo emojis/frases genéricas -> sin_categoria.

## Tarea 3 — Origen colombiano
- si | no | no_se
Reglas de desempate:
  R1. Location declarada explícita de Colombia -> si. De otro país -> no.
  R2. Sin location: bio o texto con marcadores inequívocos (instituciones,
      municipios, jerga, política local) -> si.
  R3. Marcadores ambiguos compartidos con otros países hispanos -> no_se.
  R4. Español latinoamericano genérico sin marcador -> no_se, no "si".
  R5. Se registra aparte si la LOCATION fue decisiva (uso_loc=1), para poder
      reportar una métrica secundaria excluyendo esos casos: la location es la
      misma señal que usa la supervisión a distancia del pipeline.

### Reglas adicionales fijadas durante la anotación de entidad
  R8. Entidades estatales: lider_nacional / lider_regional se reservan para
      liderazgo POLÍTICO (presidencia, ministerios, congreso, alcaldías,
      gobernaciones, concejos y sus cuentas oficiales). Agencias técnicas o
      administrativas del Estado sin rol político (INVIAS, ANSV, servicios
      fitosanitarios, defensorías de espacio público, programas ministeriales)
      -> sin_categoria.
  R9. Hospitales y redes públicas (ESE), secretarías de salud y organismos con
      mandato sanitario -> autoridad_salud, en cualquier nivel territorial.
  R10. Fuerzas militares y policía -> sin_categoria (no hay categoría iCARE).
  R11. ONG: las que trabajan directamente con comunidades -> lider_comunitario;
      las de incidencia temática (ambiental, gremial) o cuya bio no permite
      saberlo -> sin_categoria.
  R12. Universidades, facultades e institutos de investigación -> ciencia.
  R13. Nivel territorial de las cuentas de gobierno: alcaldías, gobernaciones,
      sus secretarías y dependencias de línea -> lider_regional. Ministerios,
      presidencia, congreso y sus miembros -> lider_nacional. Establecimientos
      públicos e institutos descentralizados con personería propia (institutos
      de turismo y cultura, defensorías del espacio público, ICBF, INVIAS,
      ANSV, SENA, unidades de servicios públicos, contralorías) -> sin_categoria,
      salvo mandato sanitario, que va a autoridad_salud.
  R14. Sindicatos y juntas de acción comunal -> lider_comunitario.
  R15. 'oms' es exclusivamente la OMS/OPS. Otras agencias de Naciones Unidas
      (FAO, UNICEF) no tienen categoría iCARE -> sin_categoria.
  R16. Instituciones que prestan o administran servicios de salud (hospitales,
      ESE, IPS, EPS) -> autoridad_salud. ONG cuya misión es la atención médica
      directa (p. ej. Médicos Sin Fronteras) -> profesional_salud.
