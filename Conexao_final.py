import carla
import time
import random
import threading
import numpy as np
import networkx as nx


# ---------------------------------------------------------------------------
# HEURÍSTICA A* — distância euclidiana 3D entre waypoints
# ---------------------------------------------------------------------------
def dist_heuristica(n1, n2, locations):
    p1, p2 = locations[n1], locations[n2]
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)


# ---------------------------------------------------------------------------
# ESTADO GLOBAL DE OBSTÁCULOS (thread-safe)
# ---------------------------------------------------------------------------
class EstadoObstaculo:
    def __init__(self):
        self._lock = threading.Lock()
        self.obstaculo_frontal = False
        self.faixa_esquerda_livre = True

    def atualizar(self, frontal: bool, esq_livre: bool):
        with self._lock:
            self.obstaculo_frontal = frontal
            self.faixa_esquerda_livre = esq_livre

    def ler(self):
        with self._lock:
            return self.obstaculo_frontal, self.faixa_esquerda_livre


estado = EstadoObstaculo()


# ---------------------------------------------------------------------------
# CALLBACK DO LIDAR — só grava estado, NÃO controla o veículo
# ---------------------------------------------------------------------------
def lidar_callback(data):
    points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)

    # Obstáculo frontal: pontos à frente, na altura de um carro
    frontal = points[
        (points[:, 0] > 1.5) & (points[:, 0] < 12.0) &
        (np.abs(points[:, 1]) < 1.0) &
        (points[:, 2] > -1.2)
    ]

    # Faixa esquerda: mesma janela longitudinal, offset lateral
    esq = points[
        (points[:, 0] > -2.0) & (points[:, 0] < 10.0) &
        (points[:, 1] > 1.8) & (points[:, 1] < 4.5) &
        (points[:, 2] > -1.0)
    ]

    estado.atualizar(
        frontal=len(frontal) > 20,
        esq_livre=len(esq) < 3
    )


# ---------------------------------------------------------------------------
# MONTA GRAFO A PARTIR DA TOPOLOGIA DO MAPA
# ---------------------------------------------------------------------------
def construir_grafo(carla_map):
    print("Construindo grafo de navegação...")
    graph = nx.DiGraph()
    node_locs = {}

    for seg in carla_map.get_topology():
        w1, w2 = seg[0], seg[1]
        node_locs[w1.id] = w1.transform.location
        node_locs[w2.id] = w2.transform.location
        peso = w1.transform.location.distance(w2.transform.location)
        graph.add_edge(w1.id, w2.id, weight=peso)

    print(f"  Grafo: {graph.number_of_nodes()} nós, {graph.number_of_edges()} arestas")
    return graph, node_locs


# ---------------------------------------------------------------------------
# NÓ MAIS PRÓXIMO DE UMA LOCALIZAÇÃO
# ---------------------------------------------------------------------------
def no_mais_proximo(loc, node_locs):
    return min(
        node_locs,
        key=lambda n: (node_locs[n].x - loc.x)**2 +
                      (node_locs[n].y - loc.y)**2 +
                      (node_locs[n].z - loc.z)**2
    )


# ---------------------------------------------------------------------------
# PLANEIA ROTA COM A*
# ---------------------------------------------------------------------------
def planear_rota(graph, node_locs, origem_loc, destino_loc):
    no_orig = no_mais_proximo(origem_loc, node_locs)
    no_dest = no_mais_proximo(destino_loc, node_locs)

    print(f"A*: a planear de nó {no_orig} → {no_dest} ...")
    try:
        rota = nx.astar_path(
            graph,
            no_orig,
            no_dest,
            heuristic=lambda a, b: dist_heuristica(a, b, node_locs),
            weight='weight'
        )
        custo = nx.astar_path_length(
            graph,
            no_orig,
            no_dest,
            heuristic=lambda a, b: dist_heuristica(a, b, node_locs),
            weight='weight'
        )
        print(f"  Rota encontrada: {len(rota)} waypoints, custo total = {custo:.1f} m")
        return rota
    except nx.NetworkXNoPath:
        print("  [AVISO] A* não encontrou caminho — sem destino traçado.")
        return []


# ---------------------------------------------------------------------------
# CONVERTE LISTA DE NÓS EM WAYPOINTS CARLA E DESENHA NO MAPA
# ---------------------------------------------------------------------------
def desenhar_rota(world, rota, node_locs, cor=carla.Color(0, 200, 0), duracao=60.0):
    for i in range(len(rota) - 1):
        p1 = node_locs[rota[i]]
        p2 = node_locs[rota[i + 1]]
        world.debug.draw_line(
            carla.Location(p1.x, p1.y, p1.z + 0.5),
            carla.Location(p2.x, p2.y, p2.z + 0.5),
            thickness=0.08,
            color=cor,
            life_time=duracao
        )


# ---------------------------------------------------------------------------
# CONTROLO BASEADO NA ROTA A* + ESTADO DO LIDAR
# ---------------------------------------------------------------------------
def seguir_rota(vehicle, rota, node_locs, tm, waypoint_idx=0):
    """
    Avança o veículo pelo próximo waypoint da rota A*.
    Devolve o índice atualizado.
    Chama force_lane_change se o Lidar detetar obstáculo com faixa livre.
    """
    if waypoint_idx >= len(rota):
        print("Rota concluída!")
        return waypoint_idx

    loc_atual = vehicle.get_location()
    alvo = node_locs[rota[waypoint_idx]]
    dist = loc_atual.distance(alvo)

    # Avança para o próximo waypoint quando chegar a menos de 8 m
    if dist < 8.0:
        waypoint_idx += 1
        if waypoint_idx < len(rota):
            prox = node_locs[rota[waypoint_idx]]
            print(f"  Waypoint {waypoint_idx}/{len(rota)-1} | próx alvo: "
                  f"({prox.x:.1f}, {prox.y:.1f})")

    # Reação ao Lidar
    obstaculo, esq_livre = estado.ler()
    if obstaculo:
        if esq_livre:
            print("A*: obstáculo frontal — faixa esquerda livre, a ultrapassar...")
            tm.force_lane_change(vehicle, True)
        else:
            print("A*: obstáculo frontal — faixa bloqueada, a travar...")
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.5))

    return waypoint_idx


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    # 1. Conexão
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    carla_map = world.get_map()
    bpl = world.get_blueprint_library()

    # 2. Grafo para A*
    graph, node_locs = construir_grafo(carla_map)

    # 3. Spawn do veículo
    bp = bpl.filter("model3")[0]
    spawn_points = carla_map.get_spawn_points()
    random.shuffle(spawn_points)

    vehicle = None
    for sp in spawn_points:
        vehicle = world.try_spawn_actor(bp, sp)
        if vehicle:
            print(f"Veículo gerado em: {sp.location}")
            break

    if vehicle is None:
        print("Erro: nenhum ponto de spawn disponível.")
        return

    # 4. Escolhe destino: um spawn aleatório diferente do inicial
    origem_loc = vehicle.get_location()
    candidatos = [s.location for s in spawn_points if s.location.distance(origem_loc) > 100]
    destino_loc = random.choice(candidatos) if candidatos else spawn_points[-1].location
    print(f"Destino: ({destino_loc.x:.1f}, {destino_loc.y:.1f}, {destino_loc.z:.1f})")

    # Marca o destino no mapa
    world.debug.draw_point(
        carla.Location(destino_loc.x, destino_loc.y, destino_loc.z + 1.0),
        size=0.3,
        color=carla.Color(255, 0, 0),
        life_time=120.0
    )

    # 5. Planeia rota com A*
    rota = planear_rota(graph, node_locs, origem_loc, destino_loc)
    if rota:
        desenhar_rota(world, rota, node_locs)

    # 6. Traffic Manager — piloto automático com os parâmetros de rota
    tm = client.get_trafficmanager(8000)
    vehicle.set_autopilot(True, tm.get_port())
    tm.global_percentage_speed_difference(40.0)   # 40% mais lento que o limite
    tm.set_global_distance_to_leading_vehicle(3.0)
    tm.auto_lane_change(vehicle, False)            # troca de faixa gerida pelo Lidar

    # Define rota no TM a partir dos waypoints A*
    if rota:
        waypoints_carla = []
        for no in rota:
            loc = node_locs[no]
            wp = carla_map.get_waypoint(loc, project_to_road=True)
            if wp:
                waypoints_carla.append(wp)
        if waypoints_carla:
            tm.set_path(vehicle, waypoints_carla)
            print(f"  Rota A* enviada ao TM: {len(waypoints_carla)} waypoints")

    # 7. Sensor Lidar
    lidar_bp = bpl.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '30')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('points_per_second', '56000')
    lidar_bp.set_attribute('rotation_frequency', '25')
    lidar_bp.set_attribute('upper_fov', '2.0')
    lidar_bp.set_attribute('lower_fov', '-24.8')

    lidar_sensor = world.spawn_actor(
        lidar_bp,
        carla.Transform(carla.Location(x=1.6, z=1.7)),
        attach_to=vehicle
    )
    lidar_sensor.listen(lidar_callback)

    # 8. Spectator (câmera de perseguição)
    spectator = world.get_spectator()

    # 9. Loop principal
    waypoint_idx = 0
    try:
        print("\n--- NAVEGAÇÃO A* ATIVA (Ctrl+C para parar) ---")
        while True:
            # Atualiza posição do spectator
            vt = vehicle.get_transform()
            fwd = vt.get_forward_vector()
            cam_loc = vt.location - fwd * 12 + carla.Location(z=5)
            cam_rot = vt.rotation
            cam_rot.pitch = -20
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))

            # Segue a rota A*
            waypoint_idx = seguir_rota(vehicle, rota, node_locs, tm, waypoint_idx)

            # Verifica se chegou ao destino
            dist_dest = vehicle.get_location().distance(destino_loc)
            if dist_dest < 15.0:
                print(f"\nDestino alcançado! (distância final: {dist_dest:.1f} m)")
                break

            time.sleep(0.05)   # 20 Hz é suficiente para controlo

    except KeyboardInterrupt:
        print("\nInterrompido pelo utilizador.")
    finally:
        print("A limpar atores...")
        lidar_sensor.destroy()
        vehicle.destroy()
        print("Recursos libertados.")


if __name__ == '__main__':
    main()
