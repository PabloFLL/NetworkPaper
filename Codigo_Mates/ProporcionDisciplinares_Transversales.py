import csv
import os
import networkx as nx
import numpy as np


# ============================================================
# CONFIGURACIÓN
# ============================================================

archivo_csv = r"C:\Users\pablo\Desktop\PAPERS\Redes\redmates.csv"

NODOS_TRANSVERSALES = [
    "Técnicas y estrategias de resolución de problemas relacionados con los cuerpos numéricos y estructuras",
    "Reconocimiento del error como elemento de aprendizaje en la selección u obtención de soluciones numéricas, matriciales, etc.",
    "Desarrollo histórico del sentido numérico. Aplicaciones de los conjuntos numéricos",
    "Interpretación gráfica de las soluciones de ecuaciones, inecuaciones y sistemas con y sin medios tecnológicos",
    "Desarrollo del histórico del álgebra y valoración de su uso en el avance de la ciencia y la tecnología.",
    "Flexibilidad en el uso de varias estrategias, técnicas o métodos de resolución de situaciones problemáticas susceptibles de modelación algebraica.",
    "Autonomía, tolerancia ante el error, perseverancia en el aprendizaje de aspectos asociados al sentido algebraico",
    "Resolución de problemas y modelización mediante funciones",
    "Programas informáticos de geometría dinámica. Calculadoras gráficas.",
    "Desarrollo histórico del análisis sobre funciones y sus aplicaciones. Valoración de los usos científicos de las funciones",
    "Perseverancia y flexibilidad en el cambio de estrategias, técnicas o métodos asociados a las relaciones y funciones",
    "Uso de la derivada en contextos STEM: representación gráfica, estudio del cambio y optimización.",
    "Desarrollo histórico del cálculo de integrales y derivadas, así como de sus aplicaciones.",
    "Perseverancia y flexibilidad en el cambio de estrategias, técnicas o métodos asociados al cálculo y utilización de la integral y derivada de una función.",
    "Perseverancia y flexibilidad en el cambio de estrategias, técnicas o métodos asociados al cálculo y utilización de la geometría",
    "Desarrollo histórico de la geometría analítica y sus aplicaciones. Valoración de los usos en contextos científicos",
    "Desarrollo histórico de la probabilidad y sus aplicaciones. Valoración de los usos científicos.",
    "Perseverancia y flexibilidad en el cambio de estrategias, técnicas o métodos asociados a distribuciones y el cálculo de probabilidades",
    "Estrategias de resolución de problemas. Modelización de fenómenos",
    "Calculadora, hoja de cálculo o software específico. Toma de decisiones: utilización de conclusiones derivadas del tratamiento computacional.",
    "Perseverancia, iniciativa y flexibilidad en la resolución de situaciones problemáticas susceptibles de error o no exentos de dificultades relacionados con las formas de razonamiento lógico-matemático o del uso de medios tecnológicos específicos."
]


# ============================================================
# CARGA DE RED
# ============================================================

def cargar_grafo_desde_csv(ruta_csv):

    if not os.path.exists(ruta_csv):
        print(f"Error: el archivo {ruta_csv} no existe.")
        return None, None

    with open(ruta_csv, "r", newline="", encoding="utf-8") as file:
        reader = list(csv.reader(file))

    reader = [fila for fila in reader if any(fila)]

    nombres_nodos = reader[0]

    Gtotal = nx.DiGraph()
    Gtotalsimple = nx.Graph()

    for nombre in nombres_nodos[1:]:

        Gtotal.add_node(nombre)
        Gtotalsimple.add_node(nombre)

    for i, fila in enumerate(reader[1:], start=1):

        nodo_origen = nombres_nodos[i]

        for j, valor in enumerate(fila[1:], start=1):

            nodo_destino = nombres_nodos[j]

            if nodo_origen == nodo_destino or valor == "":
                continue

            try:
                valor = int(valor)

            except ValueError:
                continue

            if valor == 0:
                continue

            if valor == 1:

                Gtotal.add_edge(nodo_origen, nodo_destino)
                Gtotalsimple.add_edge(nodo_origen, nodo_destino)

            elif valor == 2:

                Gtotal.add_edge(nodo_destino, nodo_origen)
                Gtotalsimple.add_edge(nodo_destino, nodo_origen)

            elif valor == 3:

                Gtotal.add_edge(nodo_origen, nodo_destino)
                Gtotal.add_edge(nodo_destino, nodo_origen)

                Gtotalsimple.add_edge(nodo_origen, nodo_destino)

    return Gtotal, Gtotalsimple


# ============================================================
# CLASIFICACIÓN
# ============================================================

def clasificar_nodo(nodo, nodos_transversales):

    return (
        "No disciplinar"
        if nodo in nodos_transversales
        else "Disciplinar"
    )


def comprobar_transversales(G, nodos_transversales):

    presentes = [n for n in nodos_transversales if n in G]

    ausentes = [n for n in nodos_transversales if n not in G]

    print("\n=== COMPROBACIÓN DE NODOS NO DISCIPLINARES ===")

    print("No disciplinares listados:", len(nodos_transversales))
    print("No disciplinares presentes:", len(presentes))
    print("No disciplinares no encontrados:", len(ausentes))

    return presentes, ausentes


# ============================================================
# FUNCIONES ESTADÍSTICAS
# ============================================================

def media_valores(diccionario, nodos):

    if not nodos:
        return np.nan

    return float(np.mean([diccionario[n] for n in nodos]))


def desviacion_valores(diccionario, nodos):

    if not nodos or len(nodos) < 2:
        return np.nan

    return float(
        np.std(
            [diccionario[n] for n in nodos],
            ddof=1
        )
    )


def formatear_media_sd(media, sd):

    if np.isnan(media):
        return "NA"

    if np.isnan(sd):
        return f"{media:.6f} ± NA"

    return f"{media:.6f} ± {sd:.6f}"


# ============================================================
# PRIMER CUARTIL
# ============================================================

def analizar_primer_cuartil(
        G,
        nodos_transversales,
        metrica="degree"
):

    if metrica == "degree":

        valores = dict(G.degree())

    elif metrica == "betweenness":

        valores = nx.betweenness_centrality(
            G,
            normalized=True
        )

    else:
        raise ValueError(
            "La métrica debe ser 'degree' o 'betweenness'."
        )

    nodos_ordenados = sorted(
        valores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    n = len(nodos_ordenados)

    q1_size = max(1, int(np.ceil(n * 0.25)))

    primer_cuartil = nodos_ordenados[:q1_size]

    no_disc_q1 = [
        nodo for nodo, _ in primer_cuartil
        if nodo in nodos_transversales
    ]

    disc_q1 = [
        nodo for nodo, _ in primer_cuartil
        if nodo not in nodos_transversales
    ]

    print(f"\n=== PRIMER CUARTIL SEGÚN {metrica.upper()} ===")

    print("Número total de nodos:", n)
    print("Tamaño del primer cuartil:", q1_size)

    print(
        "\nNo disciplinares en Q1:",
        len(no_disc_q1),
        f"({len(no_disc_q1) / q1_size:.2%})"
    )

    print(
        "Disciplinares en Q1:",
        len(disc_q1),
        f"({len(disc_q1) / q1_size:.2%})"
    )

    print("\nNodos del primer cuartil:")

    for nodo, valor in primer_cuartil:

        tipo = clasificar_nodo(
            nodo,
            nodos_transversales
        )

        print(
            f"{tipo:18s} | "
            f"{valor:.6f} | "
            f"{nodo}"
        )

    return primer_cuartil


# ============================================================
# BETWEENNESS POR GRUPOS
# ============================================================

def analizar_betweenness_por_grupos(
        G,
        nodos_transversales,
        criterio_orden="degree"
):

    bet = nx.betweenness_centrality(
        G,
        normalized=True
    )

    if criterio_orden == "degree":

        criterio = dict(G.degree())

    elif criterio_orden == "betweenness":

        criterio = bet

    else:

        raise ValueError(
            "criterio_orden debe ser "
            "'degree' o 'betweenness'."
        )

    nodos_ordenados = sorted(
        criterio.items(),
        key=lambda x: x[1],
        reverse=True
    )

    n = len(nodos_ordenados)

    q_size = max(1, int(np.ceil(n * 0.25)))

    top_25 = [
        nodo for nodo, _ in nodos_ordenados[:q_size]
    ]

    bottom_25 = [
        nodo for nodo, _ in nodos_ordenados[-q_size:]
    ]

    no_disc = [
        n for n in G.nodes()
        if n in nodos_transversales
    ]

    disc = [
        n for n in G.nodes()
        if n not in nodos_transversales
    ]

    top_no_disc = [
        n for n in top_25
        if n in nodos_transversales
    ]

    top_disc = [
        n for n in top_25
        if n not in nodos_transversales
    ]

    bottom_no_disc = [
        n for n in bottom_25
        if n in nodos_transversales
    ]

    bottom_disc = [
        n for n in bottom_25
        if n not in nodos_transversales
    ]

    print("\n=== BETWEENNESS POR GRUPOS ===")

    print("Criterio de orden:", criterio_orden)
    print("Tamaño cuartil:", q_size)

    print("\n--- Comparación global ---")

    print(
        "No disciplinares:",
        formatear_media_sd(
            media_valores(bet, no_disc),
            desviacion_valores(bet, no_disc)
        )
    )

    print(
        "Disciplinares:",
        formatear_media_sd(
            media_valores(bet, disc),
            desviacion_valores(bet, disc)
        )
    )

    print("\nNodos top 25%:")

    for nodo in top_25:

        tipo = clasificar_nodo(
            nodo,
            nodos_transversales
        )

        print(
            f"{tipo:18s} | "
            f"betweenness={bet[nodo]:.6f} | "
            f"criterio={criterio[nodo]:.6f} | "
            f"{nodo}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    Gtotal, Gtotalsimple = cargar_grafo_desde_csv(
        archivo_csv
    )

    if Gtotal is None:
        return

    G = Gtotalsimple

    print("=== RED CARGADA ===")

    print("Nodos:", G.number_of_nodes())
    print("Enlaces:", G.number_of_edges())

    nodos_presentes, _ = comprobar_transversales(
        G,
        NODOS_TRANSVERSALES
    )

    analizar_primer_cuartil(
        G,
        nodos_presentes,
        metrica="degree"
    )

    analizar_primer_cuartil(
        G,
        nodos_presentes,
        metrica="betweenness"
    )

    analizar_betweenness_por_grupos(
        G,
        nodos_presentes,
        criterio_orden="degree"
    )

    analizar_betweenness_por_grupos(
        G,
        nodos_presentes,
        criterio_orden="betweenness"
    )


if __name__ == "__main__":
    main()