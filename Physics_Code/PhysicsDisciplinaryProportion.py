import csv
import os
import networkx as nx
import numpy as np


# ============================================================
# CONFIGURACIÓN
# ============================================================

archivo_csv = r"C:\Users\pablo\Desktop\PAPERS\Redes\redfisica.csv"

NODOS_TRANSVERSALES = [
    "Situaciones y contextos naturales en los cuales se ponen de manifiesto diferentes fenómenos ondulatorios. Interferencias y difracción. Aplicaciones. Cambios en las propiedades de las ondas en función del desplazamiento del emisor y receptor.",
    "Aplicaciones de la óptica geométrica",
    "Papel de la física cuántica en aplicaciones como el láser, resonancias magnéticas o nanotecnología.",
    "Otras aplicaciones en los campos de la ingeniería, la tecnología y la salud.",
    "Controversias históricas originadas por la naturaleza de la materia y la energía, derivadas de la dualidad onda-corpúsculo en la luz.",
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

    if ausentes:

        print("\nNodos no encontrados:")

        for n in ausentes:
            print("-", repr(n))

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

    # ========================================================
    # TOP VS BOTTOM
    # ========================================================

    print("\n--- Comparación top/bottom ---")

    print(
        "Top 25%:",
        formatear_media_sd(
            media_valores(bet, top_25),
            desviacion_valores(bet, top_25)
        )
    )

    print(
        "Bottom 25%:",
        formatear_media_sd(
            media_valores(bet, bottom_25),
            desviacion_valores(bet, bottom_25)
        )
    )

    # ========================================================
    # GLOBAL
    # ========================================================

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

    # ========================================================
    # TOP 25
    # ========================================================

    print("\n--- Dentro del top 25% ---")

    print(
        "No disciplinares top:",
        formatear_media_sd(
            media_valores(bet, top_no_disc),
            desviacion_valores(bet, top_no_disc)
        )
    )

    print(
        "Disciplinares top:",
        formatear_media_sd(
            media_valores(bet, top_disc),
            desviacion_valores(bet, top_disc)
        )
    )

    # ========================================================
    # BOTTOM 25
    # ========================================================

    print("\n--- Dentro del bottom 25% ---")

    print(
        "No disciplinares bottom:",
        formatear_media_sd(
            media_valores(bet, bottom_no_disc),
            desviacion_valores(bet, bottom_no_disc)
        )
    )

    print(
        "Disciplinares bottom:",
        formatear_media_sd(
            media_valores(bet, bottom_disc),
            desviacion_valores(bet, bottom_disc)
        )
    )

    # ========================================================
    # NODOS TOP
    # ========================================================

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

    # Red no dirigida para centralidad
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
