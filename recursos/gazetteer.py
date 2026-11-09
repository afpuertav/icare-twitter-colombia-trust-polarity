# -*- coding: utf-8 -*-
"""
gazetteer.py — recursos lexicos para la identificacion de origen colombiano.

Estructurado en DOS NIVELES, y esa estructura es una correccion, no un
detalle de organizacion.

Por que dos niveles
-------------------
La version 1 usaba una lista plana de ~1,500 terminos: pais,
departamentos, capitales, ciudades, pueblos, parques nacionales y los
1,100+ municipios, todos con el mismo peso. Al ejecutarla sobre este
corpus, los terminos MAS activados fueron:

    colombia 191 · madrid 102 · bogota 51 · santiago 30 · california 28
    cucuta 26 · florida 26 · fundacion 25 · cali 24 · une 24 · cesar 24
    toca 22 · la paz 21 · la union 21 · cordoba 20 · valparaiso 20
    san francisco 19 · gonzalez 18 · andalucia 18 · ortega 16

De los veinte primeros, solo cinco son inequivocamente colombianos. El
resto son palabras comunes del espanol ("fundacion", "une", "cesar",
"toca"), apellidos que tambien son municipios ("gonzalez", "ortega") y
topónimos cuyo referente dominante esta fuera de Colombia ("madrid",
"santiago", "california", "florida", "valparaiso", "andalucia").

En un corpus de COVID-19 en espanol de toda Hispanoamerica, esa lista
plana marca como colombiano casi cualquier cosa.

NUCLEO (alta precision)
    Pais, departamentos, capitales y ciudades principales, quitando los
    ambiguos. Una coincidencia aqui es evidencia por si sola.

EXTENDIDO (evidencia debil)
    Municipios, pueblos patrimonio y parques nacionales. Son miles de
    nombres, muchos homonimos de palabras corrientes o de lugares de
    otros paises. Una coincidencia aqui NO basta: la etapa 03a exige que
    venga acompanada de otra senal (nucleo, jerga o Location = 'CO').

Ambas listas se filtran ademas contra las exclusiones declaradas en
config.py y, en el nivel extendido, contra una longitud minima que
descarta los homonimos cortisimos ("une", "toca", "iza", "cota", "gama").

Las listas de exclusion NO son definitivas. La etapa 03a emite
resultados/tablas/s03a_terminos_mas_frecuentes.csv precisamente para que
puedan ampliarse con evidencia del corpus completo.

JERGA
    Marcadores lexicos del espanol colombiano. Mismo problema: "china",
    "chino" y "chinos" (que en Colombia significan "nino") sumaron 396
    activaciones en la muestra, casi todas referidas a China el pais.
"""


# --- NUCLEO: alta precision ------------------------------------------------

PAIS = [
    "colombia",
]

DEPARTAMENTOS = [
    "amazonas", "antioquia", "arauca", "atlantico", "bolivar", "boyaca", "caldas",
    "caqueta", "casanare", "cauca", "cesar", "choco", "cordoba", "cundinamarca", "guainia",
    "guaviare", "huila", "la guajira", "magdalena", "meta", "nariño", "narino",
    "norte de santander", "putumayo", "quindio", "risaralda", "san andres", "santander",
    "sucre", "tolima", "valle del cauca", "vaupes", "vichada",
]

CAPITALES = [
    "leticia", "medellin", "arauca", "barranquilla", "cartagena", "tunja", "manizales",
    "florencia", "yopal", "popayan", "valledupar", "quibdo", "monteria", "bogota",
    "inirida", "san jose del guaviare", "neiva", "riohacha", "santa marta", "villavicencio",
    "pasto", "cucuta", "mocoa", "armenia", "pereira", "san andres", "bucaramanga",
    "sincelejo", "ibague", "cali", "mitu", "puerto carreno", "puerto carreño",
]

CIUDADES_PRINCIPALES = [
    "soledad", "bello", "envigado", "palmira", "itagui", "cartago", "tulua", "maicao",
    "sabanalarga", "malambo", "puerto colombia", "soledad", "bello",
]


# --- EXTENDIDO: evidencia debil, exige corroboracion ------------------------

PUEBLOS_PATRIMONIO = [
    "santa cruz de mompox", "cienaga", "aguadas", "salamina", "barichara", "jardin",
    "villa de leyva", "guadalajara de buga", "jerico", "giron", "mongui", "honda",
    "playa de belen", "pore", "guaduas", "lorica", "santa fe de antioquia", "el socorro",
]

PARQUES_NACIONALES = [
    "tayrona", "chiribiquete", "nevados", "cocuy", "purace", "serrania de la macarena",
    "caño cristales", "cano cristales", "amacayacu", "gorgona", "utria", "tatacoa",
    "corales del rosario", "guacharos", "tuparro", "isla de salamanca", "paramillo",
    "katios", "otun quimbaya", "malpelo", "santuario de fauna y flora galeras",
    "isla de la corota", "corota", "nukak", "yariguies",
]

MUNICIPIOS = [
    "abejorral", "abrego", "abriaqui", "abriaquí", "acacias", "acacías", "acandi", "acandí",
    "acevedo", "achi", "achí", "agrado", "agua de dios", "aguachica", "aguada", "aguadas",
    "aguazul", "agustin codazzi", "agustín codazzi", "aipe", "alban", "albania", "albán",
    "alcala", "alcalá", "aldana", "alejandria", "alejandría", "algarrobo", "algeciras",
    "almaguer", "almeida", "alpujarra", "altamira", "alto baudo", "alto baudó",
    "altos del rosario", "alvarado", "amaga", "amagá", "amalfi", "ambalema", "anapoima",
    "ancuya", "andalucia", "andalucía", "andes", "angelopolis", "angelópolis", "angostura",
    "anolaima", "anori", "anorí", "anserma", "ansermanuevo", "anza", "anzoategui",
    "anzoátegui", "anzá", "apartado", "apartadó", "apia", "apulo", "apía", "aquitania",
    "aracataca", "aranzazu", "aratoca", "arauca", "arauquita", "arbelaez", "arbeláez",
    "arboleda", "arboledas", "arboletes", "arcabuco", "arenal", "argelia", "ariguani",
    "ariguaní", "arjona", "armenia", "armero", "arroyohondo", "astrea", "ataco", "ayapel",
    "bagado", "bagadó", "bahia solano", "bahía solano", "bajo baudo", "bajo baudó",
    "balboa", "baranoa", "baraya", "barbacoas", "barbosa", "barichara", "barranca de upia",
    "barranca de upía", "barrancabermeja", "barrancas", "barranco de loba", "barrancominas",
    "barranquilla", "becerril", "belalcazar", "belalcázar", "belen", "belen de bajira",
    "belen de los andaquies", "belen de umbria", "bello", "belmira", "beltran", "beltrán",
    "belén", "belén de bajirá", "belén de los andaquíes", "belén de umbría", "berbeo",
    "betania", "beteitiva", "betulia", "betéitiva", "bituima", "boavita", "bochalema",
    "bogota", "bogotá", "bojaca", "bojacá", "bojaya", "bojayá", "bolivar", "bolívar",
    "bosconia", "boyaca", "boyacá", "briceno", "briceño", "bucaramanga", "bucarasica",
    "buenaventura", "buenavista", "buenos aires", "buesaco", "buga", "bugalagrande",
    "buritica", "buriticá", "busbanza", "busbanzá", "cabrera", "cabuyaro", "caceres",
    "cachipay", "cachira", "cacota", "caicedo", "caicedonia", "caimito", "cajamarca",
    "cajibio", "cajibío", "cajica", "cajicá", "calamar", "calarca", "calarcá", "caldas",
    "caldono", "cali", "california", "calima", "caloto", "campamento", "campo de la cruz",
    "campoalegre", "campohermoso", "canalete", "canasgordas", "candelaria", "cantagallo",
    "canton de san pablo", "cantón de san pablo", "caparrapi", "caparrapí", "capitanejo",
    "caqueza", "caracoli", "caracolí", "caramanta", "carcasi", "carcasí", "carepa",
    "carmen de apicala", "carmen de apicalá", "carmen de carupa", "carolina del principe",
    "carolina del príncipe", "cartagena de indias", "cartagena del chaira",
    "cartagena del chairá", "cartago", "caruru", "carurú", "casabianca",
    "castilla la nueva", "caucasia", "cañasgordas", "cepita", "cepitá", "cerete", "cereté",
    "cerinza", "cerrito", "cerro de san antonio", "certegui", "chachagui", "chachagüí",
    "chaguani", "chaguaní", "chalan", "chalán", "chameza", "chaparral", "charala",
    "charalá", "charta", "chia", "chibolo", "chigorodo", "chigorodó", "chima",
    "chimichagua", "chimá", "chinacota", "chinavita", "chinchina", "chinchiná", "chinu",
    "chinácota", "chinú", "chipaque", "chipata", "chipatá", "chiquinquira", "chiquinquirá",
    "chiquiza", "chiriguana", "chiriguaná", "chiscas", "chita", "chitaga", "chitagá",
    "chitaraque", "chivata", "chivatá", "chivor", "choachi", "choachí", "choconta",
    "chocontá", "chámeza", "chía", "chíquiza", "cicuco", "cienaga", "cienaga de oro",
    "cienega", "cimitarra", "circasia", "cisneros", "ciénaga", "ciénaga de oro", "ciénega",
    "clemencia", "cocorna", "cocorná", "coello", "cogua", "colombia", "colon", "coloso",
    "colosó", "colón", "combita", "concepcion", "concepción", "concordia", "condoto",
    "confines", "consaca", "consacá", "contadero", "contratacion", "contratación",
    "convencion", "convención", "copacabana", "coper", "cordoba", "corinto", "coromoro",
    "corozal", "corrales", "cota", "cotorra", "covarachia", "covarachía", "covenas",
    "coveñas", "coyaima", "cravo norte", "cuaspud", "cubara", "cubarral", "cubará",
    "cucaita", "cucunuba", "cucunubá", "cucuta", "cucutilla", "cuitiva", "cumaral",
    "cumaribo", "cumbal", "cumbitara", "cunday", "curillo", "curiti", "curití", "curumani",
    "curumaní", "cuítiva", "cáceres", "cáchira", "cácota", "cáqueza", "cértegui", "cómbita",
    "córdoba", "cúcuta", "dabeiba", "dagua", "dibulla", "distraccion", "distracción",
    "dolores", "donmatias", "donmatías", "dosquebradas", "duitama", "durania", "duranía",
    "ebejico", "ebéjico", "el aguila", "el atrato", "el bagre", "el banco", "el cairo",
    "el calvario", "el carmen", "el carmen de atrato", "el carmen de bolivar",
    "el carmen de bolívar", "el carmen de chucuri", "el carmen de chucurí",
    "el carmen de viboral", "el carmen del darien", "el carmen del darién", "el castillo",
    "el cerrito", "el charco", "el cocuy", "el colegio", "el copey", "el doncello",
    "el dorado", "el dovio", "el espinal", "el espino", "el guacamayo", "el guamo",
    "el molino", "el paso", "el paujil", "el penol", "el penon", "el peñol", "el peñón",
    "el pinon", "el pital", "el piñón", "el playon", "el playón", "el reten", "el retiro",
    "el retorno", "el retén", "el roble", "el rosal", "el rosario", "el santuario",
    "el socorro", "el tablon", "el tablón", "el tambo", "el tarra", "el zulia", "el águila",
    "elias", "elías", "encino", "entrerrios", "entrerríos", "envigado", "facatativa",
    "facatativá", "falan", "falán", "filadelfia", "filandia", "firavitoba", "flandes",
    "florencia", "floresta", "florian", "florida", "floridablanca", "florián", "fomeque",
    "fonseca", "fortul", "fosca", "francisco pizarro", "fredonia", "fresno", "frontino",
    "fuente de oro", "fundacion", "fundación", "funes", "funza", "fuquene", "fusagasuga",
    "fusagasugá", "fómeque", "fúquene", "gachala", "gachalá", "gachancipa", "gachancipá",
    "gachantiva", "gachantivá", "gacheta", "gachetá", "galan", "galapa", "galeras", "galán",
    "gama", "gamarra", "gambita", "gameza", "garagoa", "garzon", "garzón", "genova",
    "gigante", "ginebra", "giraldo", "girardot", "girardota", "giron", "girón",
    "gomez plata", "gonzalez", "gonzález", "gramalote", "granada", "guaca", "guacamayas",
    "guacari", "guacarí", "guachene", "guachené", "guacheta", "guachetá", "guachucal",
    "guadalupe", "guaduas", "guaitarilla", "gualmatan", "gualmatán", "guamal", "guamo",
    "guapi", "guapota", "guapotá", "guapí", "guaranda", "guarne", "guasca", "guatape",
    "guatapé", "guataqui", "guataquí", "guatavita", "guateque", "guatica", "guavata",
    "guavatá", "guayabal de siquima", "guayabal de síquima", "guayabetal", "guayata",
    "guayatá", "guepsa", "guican", "gutierrez", "gutiérrez", "guática", "gámbita", "gámeza",
    "génova", "gómez plata", "güepsa", "güicán", "hacari", "hacarí", "hatillo de loba",
    "hato", "hato corozal", "hatonuevo", "heliconia", "herran", "herrán", "herveo",
    "hispania", "hobo", "honda", "ibague", "ibagué", "icononzo", "iles", "imues", "imués",
    "inirida", "inza", "inzá", "inírida", "ipiales", "iquira", "isnos", "istmina", "itagui",
    "itagüí", "ituango", "iza", "jambalo", "jambaló", "jamundi", "jamundí", "jardin",
    "jardín", "jenesano", "jerico", "jericó", "jerusalen", "jerusalén", "jesus maria",
    "jesús maría", "jordan", "jordán", "juan de acosta", "junin", "junín", "jurado",
    "juradó", "la apartada", "la argentina", "la belleza", "la calera", "la capilla",
    "la ceja", "la celia", "la cruz", "la cumbre", "la dorada", "la esperanza",
    "la estrella", "la florida", "la gloria", "la jagua de ibirico", "la jagua del pilar",
    "la llanada", "la macarena", "la merced", "la mesa", "la montanita", "la montañita",
    "la palma", "la paz", "la pena", "la peña", "la pintada", "la plata",
    "la playa de belen", "la playa de belén", "la primavera", "la salina", "la sierra",
    "la tebaida", "la tola", "la union", "la unión", "la uribe", "la uvita", "la vega",
    "la victoria", "la virginia", "labateca", "labranzagrande", "landazuri", "landázuri",
    "lebrija", "leiva", "lejanias", "lejanías", "lenguazaque", "lerida", "leticia",
    "libano", "liborina", "linares", "litoral de san juan", "lloro", "lloró",
    "lopez de micay", "lorica", "los andes", "los cordobas", "los córdobas", "los palmitos",
    "los patios", "los santos", "lourdes", "luruaco", "lérida", "líbano", "lópez de micay",
    "macanal", "macaravita", "maceo", "macheta", "machetá", "madrid", "magangue",
    "magangué", "magui payan", "magüí payán", "mahates", "maicao", "majagual", "malaga",
    "malambo", "mallama", "manati", "manatí", "manaure", "manaure balcon del cesar",
    "manaure balcón del cesar", "mani", "manizales", "manta", "manzanares", "maní",
    "mapiripan", "mapiripán", "margarita", "maria la baja", "marinilla", "maripi", "maripí",
    "mariquita", "marmato", "marquetalia", "marsella", "marulanda", "maría la baja",
    "matanza", "medellin", "medellín", "medina", "medio atrato", "medio baudo",
    "medio baudó", "medio san juan", "melgar", "mercaderes", "mesetas", "miraflores",
    "miranda", "mistrato", "mistrató", "mitu", "mitú", "mocoa", "mogotes", "molagavita",
    "momil", "mompos", "mompós", "mongua", "mongui", "monguí", "moniquira", "moniquirá",
    "monitos", "montebello", "montecristo", "montelibano", "montelíbano", "montenegro",
    "monteria", "monterrey", "montería", "morales", "morelia", "morroa", "mosquera",
    "motavita", "moñitos", "murillo", "murindo", "murindó", "mutata", "mutatá", "mutiscua",
    "muzo", "málaga", "narino", "nariño", "nataga", "natagaima", "nechi", "nechí",
    "necocli", "necoclí", "neira", "neiva", "nemocon", "nemocón", "nilo", "nimaima",
    "nobsa", "nocaima", "norcasia", "norosi", "norosí", "novita", "nueva granada",
    "nuevo colon", "nuevo colón", "nunchia", "nunchía", "nuqui", "nuquí", "nátaga",
    "nóvita", "obando", "ocamonte", "ocana", "ocaña", "oiba", "oicata", "oicatá", "olaya",
    "olaya herrera", "onzaga", "oporapa", "orito", "orocue", "orocué", "ortega", "ospina",
    "otanche", "ovejas", "pachavita", "pacho", "pacora", "padilla", "paez", "paicol",
    "pailitas", "paime", "paipa", "pajarito", "palermo", "palestina", "palmar",
    "palmar de varela", "palmas del socorro", "palmira", "palocabildo", "pamplona",
    "pamplonita", "pandi", "panqueba", "paramo", "paratebueno", "pasca", "pasto", "patia",
    "patía", "pauna", "paya", "paz de ariporo", "paz del rio", "paz del río", "pedraza",
    "pelaya", "pensilvania", "peque", "pereira", "pesca", "piamonte", "piedecuesta",
    "piedras", "piendamo", "piendamó", "pijao", "pijino del carmen", "pijiño del carmen",
    "pinchote", "pinillos", "piojo", "piojó", "pisba", "pitalito", "pivijay", "planadas",
    "planeta rica", "plato", "policarpa", "polonuevo", "ponedera", "popayan", "popayán",
    "pore", "potosi", "potosí", "pradera", "prado", "providencia",
    "providencia y santa catalina islas", "pueblo bello", "pueblo nuevo", "pueblo rico",
    "pueblo viejo", "pueblorrico", "puente nacional", "puerres", "puerto asis",
    "puerto asís", "puerto berrio", "puerto berrío", "puerto boyaca", "puerto boyacá",
    "puerto caicedo", "puerto carreno", "puerto carreño", "puerto colombia",
    "puerto concordia", "puerto escondido", "puerto gaitan", "puerto gaitán",
    "puerto guzman", "puerto guzmán", "puerto leguizamo", "puerto leguízamo",
    "puerto libertador", "puerto lleras", "puerto lopez", "puerto lópez", "puerto milan",
    "puerto milán", "puerto nare", "puerto narino", "puerto nariño", "puerto parra",
    "puerto rico", "puerto rondon", "puerto rondón", "puerto salgar", "puerto santander",
    "puerto tejada", "puerto triunfo", "puerto wilches", "puli", "pulí", "pupiales",
    "purace", "puracé", "purificacion", "purificación", "purisima", "purísima", "pácora",
    "páez", "páramo", "quebradanegra", "quetame", "quibdo", "quibdó", "quimbaya",
    "quinchia", "quinchía", "quipama", "quipile", "quípama", "ragonvalia", "ramiriqui",
    "ramiriquí", "raquira", "recetor", "regidor", "remedios", "remolino", "repelon",
    "repelón", "restrepo", "ricaurte", "rio de oro", "rio iro", "rio quito", "rio viejo",
    "rioblanco", "riofrio", "riofrío", "riohacha", "rionegro", "riosucio", "risaralda",
    "rivera", "roberto payan", "roberto payán", "roldanillo", "roncesvalles", "rondon",
    "rondón", "rosas", "rovira", "ráquira", "río de oro", "río iró", "río quito",
    "río viejo", "sabana de torres", "sabanagrande", "sabanalarga", "sabanas de san angel",
    "sabanas de san ángel", "sabaneta", "saboya", "saboyá", "sacama", "sachica", "sahagun",
    "sahagún", "saladoblanco", "salamina", "salazar de las palmas", "saldana", "saldaña",
    "salento", "salgar", "samaca", "samacá", "samana", "samaniego", "samaná", "sampues",
    "sampués", "san agustin", "san agustín", "san alberto", "san andres",
    "san andres de cuerquia", "san andres de sotavento", "san andrés",
    "san andrés de cuerquia", "san andrés de sotavento", "san antero", "san antonio",
    "san antonio de palmito", "san antonio del tequendama", "san benito", "san benito abad",
    "san bernardo", "san bernardo del viento", "san calixto", "san carlos",
    "san carlos de guaroa", "san cayetano", "san cristobal", "san cristóbal", "san diego",
    "san eduardo", "san estanislao", "san fernando", "san francisco", "san gil",
    "san jacinto", "san jacinto del cauca", "san jeronimo", "san jerónimo", "san joaquin",
    "san joaquín", "san jose", "san jose de alban", "san jose de la montana",
    "san jose de miranda", "san jose de pare", "san jose de ure", "san jose del fragua",
    "san jose del guaviare", "san jose del palmar", "san josé", "san josé de albán",
    "san josé de la montaña", "san josé de miranda", "san josé de pare", "san josé de uré",
    "san josé del fragua", "san josé del guaviare", "san josé del palmar",
    "san juan de arama", "san juan de betulia", "san juan de rioseco", "san juan de uraba",
    "san juan de urabá", "san juan del cesar", "san juan nepomuceno", "san juanito",
    "san lorenzo", "san luis", "san luis de gaceno", "san luis de palenque", "san marcos",
    "san martin", "san martin de loba", "san martín", "san martín de loba", "san mateo",
    "san miguel", "san miguel de sema", "san onofre", "san pablo", "san pablo de borbur",
    "san pedro", "san pedro de cartago", "san pedro de los milagros", "san pedro de uraba",
    "san pedro de urabá", "san pelayo", "san rafael", "san roque", "san sebastian",
    "san sebastian de buenavista", "san sebastián", "san sebastián de buenavista",
    "san vicente", "san vicente de chucuri", "san vicente de chucurí",
    "san vicente del caguan", "san vicente del caguán", "san zenon", "san zenón", "sandona",
    "sandoná", "santa ana", "santa barbara", "santa barbara de pinto", "santa bárbara",
    "santa bárbara de pinto", "santa catalina", "santa fe de antioquia",
    "santa helena del opon", "santa helena del opón", "santa isabel", "santa lucia",
    "santa lucía", "santa maria", "santa marta", "santa maría", "santa rosa",
    "santa rosa de cabal", "santa rosa de osos", "santa rosa de viterbo",
    "santa rosa del sur", "santa rosalia", "santa rosalía", "santa sofia", "santa sofía",
    "santacruz", "santana", "santander de quilichao", "santiago", "santo domingo",
    "santo domingo de silos", "santo tomas", "santo tomás", "santuario", "sapuyes",
    "saravena", "sardinata", "sasaima", "sativanorte", "sativasur", "segovia", "sesquile",
    "sesquilé", "sevilla", "siachoque", "sibate", "sibaté", "sibundoy", "silvania",
    "silvia", "simacota", "simijaca", "simiti", "simití", "since", "sincelejo", "sincé",
    "sipi", "sipí", "sitionuevo", "soacha", "soata", "soatá", "socha", "socota", "socotá",
    "sogamoso", "solano", "soledad", "solita", "somondoco", "sonson", "sonsón", "sopetran",
    "sopetrán", "soplaviento", "sopo", "sopó", "sora", "soraca", "soracá", "sotaquira",
    "sotaquirá", "sotara", "sotará", "suaita", "suan", "suarez", "suaza", "subachoque",
    "sucre", "suesca", "supata", "supatá", "supia", "supía", "surata", "suratá", "susa",
    "susacon", "susacón", "sutamarchan", "sutamarchán", "sutatausa", "sutatenza", "suán",
    "suárez", "sácama", "sáchica", "tabio", "tado", "tadó", "talaigua nuevo", "tamalameque",
    "tamara", "tame", "tamesis", "taminango", "tangua", "taraira", "taraza", "tarazá",
    "tarqui", "tarquí", "tarso", "tasco", "tauramena", "tausa", "tello", "tena", "tenerife",
    "tenjo", "tenza", "teorama", "teruel", "tesalia", "tibacuy", "tibana", "tibaná",
    "tibasosa", "tibirita", "tibu", "tibú", "tierralta", "timana", "timaná", "timbio",
    "timbiqui", "timbiquí", "timbío", "tinjaca", "tinjacá", "tipacoque", "tiquisio",
    "titiribi", "titiribí", "toca", "tocaima", "tocancipa", "tocancipá", "togui", "togüí",
    "toledo", "tolu", "tolu viejo", "tolú", "tolú viejo", "tona", "topaga", "topaipi",
    "topaipí", "toribio", "toribío", "toro", "tota", "totoro", "totoró", "trinidad",
    "trujillo", "tubara", "tubará", "tuchin", "tuchín", "tulua", "tuluá", "tumaco", "tunja",
    "tunungua", "tununguá", "tuquerres", "turbaco", "turbana", "turbaná", "turbo",
    "turmeque", "turmequé", "tuta", "tutaza", "tutazá", "támara", "támesis", "tópaga",
    "túquerres", "ubala", "ubalá", "ubaque", "ubate", "ubaté", "ulloa", "umbita", "une",
    "unguia", "unguía", "union panamericana", "unión panamericana", "uramita", "uribia",
    "urrao", "urumita", "usiacuri", "usiacurí", "utica", "valdivia", "valencia",
    "valle de san jose", "valle de san josé", "valle de san juan", "valle del guamuez",
    "valledupar", "valparaiso", "valparaíso", "vegachi", "vegachí", "velez", "venadillo",
    "venecia", "ventaquemada", "vergara", "versalles", "vetas", "viani", "vianí",
    "victoria", "vigia del fuerte", "vigía del fuerte", "vijes", "villa caro",
    "villa de leyva", "villa del rosario", "villa rica", "villagarzon", "villagarzón",
    "villagomez", "villagómez", "villahermosa", "villamaria", "villamaría", "villanueva",
    "villapinzon", "villapinzón", "villarrica", "villavicencio", "villavieja", "villeta",
    "viota", "viotá", "viracacha", "viracachá", "vista hermosa", "viterbo", "vélez",
    "yacopi", "yacopí", "yacuanquer", "yaguara", "yaguará", "yali", "yalí", "yarumal",
    "yolombo", "yolombó", "yondo", "yondó", "yopal", "yotoco", "yumbo", "zambrano",
    "zapatoca", "zapayan", "zapayán", "zaragoza", "zarzal", "zetaquira", "zetaquirá",
    "zipacon", "zipacón", "zipaquira", "zipaquirá", "zona bananera", "ábrego", "íquira",
    "úmbita", "útica",
]


# --- LEXICON DE JERGA COLOMBIANA -------------------------------------------

JERGA = [
    "cascar", "farra", "farrear", "farreando", "guaro", "guayabo", "jincho", "jincha",
    "jinchos", "jinchas", "trago", "polas", "pola", "balde", "berraquera", "berraco",
    "berraca", "berracos", "berracas", "caneca", "cachucha", "cantaleta", "cantaletoso",
    "cantaletosa", "cantaletosos", "cantaletosas", "camello", "camella", "camellas",
    "camellar", "camellando", "chaguala", "chepa", "chechere", "chiripa", "chimba",
    "chimbo", "chimbos", "chimbas", "chucha", "chuspa", "cucos", "cumbamba", "despelote",
    "drogueria", "entuque", "filo", "filos", "guevonada", "guevonadas", "aguevonado",
    "aguevonada", "aguevonados", "aguevonadas", "jartera", "jarto", "jarta", "jartos",
    "jartas", "locha", "melona", "motoso", "motosa", "motosos", "motosas", "mota", "monda",
    "cule", "paila", "pecueca", "tombo", "tomba", "tombos", "tombas", "trancon", "trola",
    "vaina", "vainas", "verguero", "achantao", "achantada", "achantaos", "achantadas",
    "atolondrado", "atolondrada", "atolondrados", "atolondradas", "bacano", "bacana",
    "bacanos", "bacanas", "bacaneria", "boleta", "canson", "cansona", "cansones",
    "cansonas", "chino", "china", "chinos", "chinas", "cucho", "cucha", "cuchos", "cuchas",
    "culicagao", "culicagada", "culicagaos", "culicagadas", "chanda", "chandoso",
    "chandosa", "chandosos", "chandosas", "chichipato", "chichipata", "chichipatos",
    "chichipatas", "ficti", "gamin", "gamina", "gamines", "gomelo", "gomela", "gomelos",
    "gomelas", "guevon", "guevona", "guevones", "guevonas", "ingrimo", "ingrima",
    "ingrimos", "ingrimas", "lacra", "mamon", "mamona", "mamonas", "mamones", "metido",
    "metida", "metidos", "metidas", "nero", "nera", "neros", "neras", "ñero", "ñera",
    "ñeros", "ñeras", "parcero", "parcera", "parceros", "parceras", "parce", "parces",
    "pelao", "pelaa", "pelaos", "pelaas", "picho", "picha", "pichos", "pichas", "pirobo",
    "piroba", "pirobos", "pirobas", "pulido", "pulida", "pulidos", "pulidas", "toche",
    "toches", "amanarse", "dar papaya", "dio papaya", "di papaya", "dimos papaya",
    "se corrio", "destutanarse", "destutuno", "encularse", "enculo", "encule",
    "embolatarse", "embolate", "embolato", "estar pilas", "hacer una vaca",
    "pedir un catorce", "manejar", "ponerse las pilas", "parar bolas", "rumbear",
    "tirar caja", "tirar los perros", "vitriniar", "atembado", "atembada", "atembados",
    "atembadas", "bergajo", "bergaja", "bergajos", "bergajas", "carranguero", "carranguera",
    "carrangueros", "carrangueras", "guache", "guaches", "guambito", "guambita",
    "guambitos", "guambitas", "mecato", "mecatos", "mekato", "transmilenio",
    "transmilenios", "parchar", "borondo", "chichigua", "cosianfiro", "cosianpirulo",
    "melo", "peye", "sumerce", "tetiado", "tetiada", "chevere",
]


# --- Agregados que consumen las etapas -------------------------------------
GAZETTEER_NUCLEO = PAIS + DEPARTAMENTOS + CAPITALES + CIUDADES_PRINCIPALES
GAZETTEER_EXTENDIDO = PUEBLOS_PATRIMONIO + PARQUES_NACIONALES + MUNICIPIOS

# Retrocompatibilidad con la version 1 (lista plana). No la usa el
# pipeline nuevo; se conserva para poder reproducir el comportamiento
# antiguo si hiciera falta compararlo.
gazetteer = GAZETTEER_NUCLEO + GAZETTEER_EXTENDIDO
jerga = JERGA
