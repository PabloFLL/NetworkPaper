import csv
import os
import random
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
# FUNCIONES AUXILIARES
# ============================================================

def clasificar_nodo(nodo, nodos_transversales):
    return "Transversal" if nodo in nodos_transversales else "Disciplinar"


def comprobar_transversales(G, nodos_transversales):
    presentes = [n for n in nodos_transversales if n in G]
    ausentes = [n for n in nodos_transversales if n not in G]

    print("\n=== COMPROBACIÓN DE NODOS TRANSVERSALES ===")
    print("Transversales listados:", len(nodos_transversales))
    print("Transversales presentes en la red:", len(presentes))
    print("Transversales no encontrados:", len(ausentes))

    if ausentes:
        print("\nNodos no encontrados:")
        for n in ausentes:
            print("-", repr(n))

    return presentes, ausentes


def media_valores(diccionario, nodos):
    if not nodos:
        return np.nan
    return float(np.mean([diccionario[n] for n in nodos]))


def obtener_valores_centralidad(G, metrica="degree"):
    if metrica == "degree":
        return dict(G.degree())
    elif metrica == "betweenness":
        return nx.betweenness_centrality(G, normalized=True)
    else:
        raise ValueError("La métrica debe ser 'degree' o 'betweenness'.")


def obtener_top_cuartil(G, metrica="degree"):
    valores = obtener_valores_centralidad(G, metrica)
    nodos_ordenados = sorted(valores.items(), key=lambda x: x[1], reverse=True)

    n = len(nodos_ordenados)
    q1_size = max(1, int(np.ceil(n * 0.25)))

    return nodos_ordenados[:q1_size], valores


# ============================================================
# ANÁLISIS 1: PROPORCIÓN EN PRIMER CUARTIL
# ============================================================

def analizar_primer_cuartil(G, nodos_transversales, metrica="degree"):
    primer_cuartil, valores = obtener_top_cuartil(G, metrica)

    n = G.number_of_nodes()
    q1_size = len(primer_cuartil)

    transversales_q1 = [
        nodo for nodo, _ in primer_cuartil
        if nodo in nodos_transversales
    ]

    disciplinares_q1 = [
        nodo for nodo, _ in primer_cuartil
        if nodo not in nodos_transversales
    ]

    print(f"\n=== PRIMER CUARTIL SEGÚN {metrica.upper()} ===")
    print("Número total de nodos:", n)
    print("Tamaño del primer cuartil:", q1_size)

    print("\nTransversales en Q1:", len(transversales_q1),
          f"({len(transversales_q1) / q1_size:.2%})")

    print("Disciplinares en Q1:", len(disciplinares_q1),
          f"({len(disciplinares_q1) / q1_size:.2%})")

    print("\nNodos del primer cuartil:")
    for nodo, valor in primer_cuartil:
        tipo = clasificar_nodo(nodo, nodos_transversales)
        print(f"{tipo:12s} | {valor:.6f} | {nodo}")

    return primer_cuartil, transversales_q1, disciplinares_q1


# ============================================================
# ANÁLISIS 2: BETWEENNESS PROMEDIO POR GRUPOS
# ============================================================

def analizar_betweenness_por_grupos(G, nodos_transversales, criterio_orden="degree"):
    bet = nx.betweenness_centrality(G, normalized=True)

    if criterio_orden == "degree":
        criterio = dict(G.degree())
    elif criterio_orden == "betweenness":
        criterio = bet
    else:
        raise ValueError("criterio_orden debe ser 'degree' o 'betweenness'.")

    nodos_ordenados = sorted(criterio.items(), key=lambda x: x[1], reverse=True)

    n = len(nodos_ordenados)
    q_size = max(1, int(np.ceil(n * 0.25)))

    top_25 = [nodo for nodo, _ in nodos_ordenados[:q_size]]
    bottom_25 = [nodo for nodo, _ in nodos_ordenados[-q_size:]]

    transversales = [n for n in G.nodes() if n in nodos_transversales]
    disciplinares = [n for n in G.nodes() if n not in nodos_transversales]

    top_trans = [n for n in top_25 if n in nodos_transversales]
    top_disc = [n for n in top_25 if n not in nodos_transversales]

    bottom_trans = [n for n in bottom_25 if n in nodos_transversales]
    bottom_disc = [n for n in bottom_25 if n not in nodos_transversales]

    print(f"\n=== BETWEENNESS PROMEDIO POR GRUPOS ===")
    print(f"Criterio para definir centralidad alta/baja: {criterio_orden}")
    print("Tamaño de cada cuartil:", q_size)

    print("\n--- Comparación top/bottom ---")
    print("Betweenness media top 25%:", media_valores(bet, top_25))
    print("Betweenness media bottom 25%:", media_valores(bet, bottom_25))

    print("\n--- Comparación transversal/disciplinar global ---")
    print("Betweenness media transversales:", media_valores(bet, transversales))
    print("Betweenness media disciplinares:", media_valores(bet, disciplinares))

    print("\n--- Dentro del top 25% ---")
    print("Betweenness media transversales top 25%:", media_valores(bet, top_trans))
    print("Betweenness media disciplinares top 25%:", media_valores(bet, top_disc))

    print("\n--- Dentro del bottom 25% ---")
    print("Betweenness media transversales bottom 25%:", media_valores(bet, bottom_trans))
    print("Betweenness media disciplinares bottom 25%:", media_valores(bet, bottom_disc))

    print("\nNodos top 25%:")
    for nodo in top_25:
        tipo = clasificar_nodo(nodo, nodos_transversales)
        print(f"{tipo:12s} | betweenness={bet[nodo]:.6f} | criterio={criterio[nodo]:.6f} | {nodo}")

    return {
        "betweenness": bet,
        "top_25": top_25,
        "bottom_25": bottom_25,
        "transversales": transversales,
        "disciplinares": disciplinares,
        "top_trans": top_trans,
        "top_disc": top_disc,
        "bottom_trans": bottom_trans,
        "bottom_disc": bottom_disc,
    }


# ============================================================
# PERMUTATION TEST 1:
# SOBRERREPRESENTACIÓN DE TRANSVERSALES EN Q1
# ============================================================

def permutation_test_proporcion_q1(
    G,
    nodos_transversales,
    metrica="degree",
    n_iter=10000,
    seed=42,
    alternative="greater"
):
    """
    Test de permutación para comprobar si hay más transversales en el Q1
    de lo esperable al azar.

    alternative:
    - "greater": prueba si hay sobrerrepresentación de transversales en Q1.
    - "two-sided": prueba si el valor observado es extremo en cualquier dirección.
    """

    random.seed(seed)
    np.random.seed(seed)

    nodes = list(G.nodes())
    trans_set = set(nodos_transversales)

    k_trans = len(trans_set)

    primer_cuartil, _ = obtener_top_cuartil(G, metrica)
    q1_nodes = set(n for n, _ in primer_cuartil)
    q1_size = len(q1_nodes)

    obs_count = len(q1_nodes & trans_set)
    obs_prop = obs_count / q1_size

    random_counts = np.empty(n_iter, dtype=int)

    for i in range(n_iter):
        random_trans = set(random.sample(nodes, k_trans))
        random_counts[i] = len(q1_nodes & random_trans)

    random_props = random_counts / q1_size

    if alternative == "greater":
        p_value = (np.sum(random_counts >= obs_count) + 1) / (n_iter + 1)
    elif alternative == "two-sided":
        mean_null = random_props.mean()
        p_value = (np.sum(np.abs(random_props - mean_null) >= abs(obs_prop - mean_null)) + 1) / (n_iter + 1)
    else:
        raise ValueError("alternative debe ser 'greater' o 'two-sided'.")

    print(f"\n=== PERMUTATION TEST Q1 ({metrica.upper()}) ===")
    print("Transversales totales:", k_trans)
    print("Tamaño Q1:", q1_size)
    print("Transversales observados en Q1:", obs_count)
    print("Proporción observada:", obs_prop)
    print("Media nula:", float(random_props.mean()))
    print("IC95% nulo:", (float(np.percentile(random_props, 2.5)),
                          float(np.percentile(random_props, 97.5))))
    print("p-valor:", p_value)

    return {
        "obs_count": obs_count,
        "obs_prop": obs_prop,
        "null_counts": random_counts,
        "null_props": random_props,
        "p_value": p_value,
    }


# ============================================================
# PERMUTATION TEST 2:
# DIFERENCIA DE BETWEENNESS TRANSVERSALES VS DISCIPLINARES
# ============================================================

def permutation_test_betweenness_global(
    G,
    nodos_transversales,
    n_iter=10000,
    seed=42,
    alternative="greater"
):
    """
    Test de permutación para comprobar si la betweenness media de los
    transversales es mayor que la de los disciplinares.
    """

    random.seed(seed)
    np.random.seed(seed)

    nodes = list(G.nodes())
    trans_set = set(nodos_transversales)
    k_trans = len(trans_set)

    bet = nx.betweenness_centrality(G, normalized=True)

    trans_real = [bet[n] for n in nodes if n in trans_set]
    disc_real = [bet[n] for n in nodes if n not in trans_set]

    diff_obs = np.mean(trans_real) - np.mean(disc_real)

    diffs_null = np.empty(n_iter, dtype=float)

    for i in range(n_iter):
        random_trans = set(random.sample(nodes, k_trans))

        trans_rand = [bet[n] for n in nodes if n in random_trans]
        disc_rand = [bet[n] for n in nodes if n not in random_trans]

        diffs_null[i] = np.mean(trans_rand) - np.mean(disc_rand)

    if alternative == "greater":
        p_value = (np.sum(diffs_null >= diff_obs) + 1) / (n_iter + 1)
    elif alternative == "two-sided":
        mean_null = diffs_null.mean()
        p_value = (np.sum(np.abs(diffs_null - mean_null) >= abs(diff_obs - mean_null)) + 1) / (n_iter + 1)
    else:
        raise ValueError("alternative debe ser 'greater' o 'two-sided'.")

    print("\n=== PERMUTATION TEST BETWEENNESS GLOBAL ===")
    print("Betweenness media transversales:", float(np.mean(trans_real)))
    print("Betweenness media disciplinares:", float(np.mean(disc_real)))
    print("Diferencia observada:", float(diff_obs))
    print("Media nula:", float(diffs_null.mean()))
    print("IC95% nulo:", (float(np.percentile(diffs_null, 2.5)),
                          float(np.percentile(diffs_null, 97.5))))
    print("p-valor:", p_value)

    return {
        "diff_obs": diff_obs,
        "diffs_null": diffs_null,
        "p_value": p_value,
    }


# ============================================================
# PERMUTATION TEST 3:
# DIFERENCIA DE BETWEENNESS DENTRO DEL TOP 25%
# ============================================================

def permutation_test_betweenness_top_q1(
    G,
    nodos_transversales,
    metrica="degree",
    n_iter=10000,
    seed=42,
    alternative="greater"
):
    """
    Test de permutación para comprobar si, dentro del Q1,
    los transversales tienen mayor betweenness media que los disciplinares.

    Nota:
    Se mantiene fijo el Q1 y se permutan las etiquetas transversal/disciplinar.
    """

    random.seed(seed)
    np.random.seed(seed)

    nodes = list(G.nodes())
    trans_set = set(nodos_transversales)
    k_trans = len(trans_set)

    bet = nx.betweenness_centrality(G, normalized=True)

    primer_cuartil, _ = obtener_top_cuartil(G, metrica)
    q1_nodes = [n for n, _ in primer_cuartil]

    q1_trans_real = [n for n in q1_nodes if n in trans_set]
    q1_disc_real = [n for n in q1_nodes if n not in trans_set]

    if len(q1_trans_real) == 0 or len(q1_disc_real) == 0:
        print(f"\n=== PERMUTATION TEST BETWEENNESS TOP Q1 ({metrica.upper()}) ===")
        print("No se puede calcular: Q1 no contiene ambos grupos.")
        return None

    diff_obs = (
        np.mean([bet[n] for n in q1_trans_real])
        - np.mean([bet[n] for n in q1_disc_real])
    )

    diffs_null = []

    for _ in range(n_iter):
        random_trans = set(random.sample(nodes, k_trans))

        q1_trans_rand = [n for n in q1_nodes if n in random_trans]
        q1_disc_rand = [n for n in q1_nodes if n not in random_trans]

        if len(q1_trans_rand) == 0 or len(q1_disc_rand) == 0:
            continue

        diff_rand = (
            np.mean([bet[n] for n in q1_trans_rand])
            - np.mean([bet[n] for n in q1_disc_rand])
        )
        diffs_null.append(diff_rand)

    diffs_null = np.array(diffs_null, dtype=float)

    if len(diffs_null) == 0:
        print("No hay permutaciones válidas.")
        return None

    if alternative == "greater":
        p_value = (np.sum(diffs_null >= diff_obs) + 1) / (len(diffs_null) + 1)
    elif alternative == "two-sided":
        mean_null = diffs_null.mean()
        p_value = (np.sum(np.abs(diffs_null - mean_null) >= abs(diff_obs - mean_null)) + 1) / (len(diffs_null) + 1)
    else:
        raise ValueError("alternative debe ser 'greater' o 'two-sided'.")

    print(f"\n=== PERMUTATION TEST BETWEENNESS TOP Q1 ({metrica.upper()}) ===")
    print("Iteraciones válidas:", len(diffs_null))
    print("Diferencia observada:", float(diff_obs))
    print("Media nula:", float(diffs_null.mean()))
    print("IC95% nulo:", (float(np.percentile(diffs_null, 2.5)),
                          float(np.percentile(diffs_null, 97.5))))
    print("p-valor:", p_value)

    return {
        "diff_obs": diff_obs,
        "diffs_null": diffs_null,
        "p_value": p_value,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    Gtotal, Gtotalsimple = cargar_grafo_desde_csv(archivo_csv)

    if Gtotal is None:
        return

    G = Gtotalsimple

    print("=== RED CARGADA ===")
    print("Nodos:", G.number_of_nodes())
    print("Enlaces:", G.number_of_edges())

    nodos_transversales_presentes, _ = comprobar_transversales(G, NODOS_TRANSVERSALES)

    analizar_primer_cuartil(
        G,
        nodos_transversales_presentes,
        metrica="degree"
    )

    analizar_primer_cuartil(
        G,
        nodos_transversales_presentes,
        metrica="betweenness"
    )

    analizar_betweenness_por_grupos(
        G,
        nodos_transversales_presentes,
        criterio_orden="degree"
    )

    analizar_betweenness_por_grupos(
        G,
        nodos_transversales_presentes,
        criterio_orden="betweenness"
    )

    # ========================================================
    # PERMUTATION TESTS
    # ========================================================

    permutation_test_proporcion_q1(
        G,
        nodos_transversales_presentes,
        metrica="degree",
        n_iter=10000,
        seed=42,
        alternative="greater"
    )

    permutation_test_proporcion_q1(
        G,
        nodos_transversales_presentes,
        metrica="betweenness",
        n_iter=10000,
        seed=42,
        alternative="greater"
    )

    permutation_test_betweenness_global(
        G,
        nodos_transversales_presentes,
        n_iter=10000,
        seed=42,
        alternative="greater"
    )

    permutation_test_betweenness_top_q1(
        G,
        nodos_transversales_presentes,
        metrica="degree",
        n_iter=10000,
        seed=42,
        alternative="greater"
    )

    permutation_test_betweenness_top_q1(
        G,
        nodos_transversales_presentes,
        metrica="betweenness",
        n_iter=10000,
        seed=42,
        alternative="greater"
    )


if __name__ == "__main__":
    main()