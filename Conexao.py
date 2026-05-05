import carla
import time
import numpy as np
import networkx as nx

def dist_heuristica(n1, n2, locations):
    """Função h(n) para o A*: Distância euclidiana entre dois nós."""
    p1 = locations[n1]
    p2 = locations[n2]
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def main():
    # 1. CONEXÃO E SETUP
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    blueprint_library = world.get_blueprint_library()

    # 2. CONSTRUÇÃO DO GRAFO DE NAVEGAÇÃO (Para o A*)
    print("Mapeando a cidade para o algoritmo A*...")
    topology = carla_map.get_topology()
    graph = nx.DiGraph()
    node_locations = {}

    for segment in topology:
        wp1, wp2 = segment[0], segment[1]
        node_locations[wp1.id] = wp1.transform.location
        node_locations[wp2.id] = wp2.transform.location
        dist = wp1.transform.location.distance(wp2.transform.location)
        graph.add_edge(wp1.id, wp2.id, weight=dist)

    # 3. SPAWN DO VEÍCULO E CÂMERA ESPECTADORA
    vehicle_bp = blueprint_library.filter("model3")[0]
    spawn_point = carla_map.get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Posicionar câmera para você ver o teste na interface
    spectator = world.get_spectator()
    spec_trans = carla.Transform(spawn_point.location + carla.Location(z=40), 
                                 carla.Rotation(pitch=-90))
    spectator.set_transform(spec_trans)

    # 4. CONFIGURAR AUTOPILOTO (Traffic Manager)
    tm = client.get_trafficmanager(8000)
    vehicle.set_autopilot(True, tm.get_port())

    # 5. CONFIGURAR SENSOR LIDAR
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '30')
    lidar_bp.set_attribute('rotation_frequency', '20')
    lidar_sensor = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=1.6, z=1.7)), attach_to=vehicle)

    # 6. LÓGICA DO AGENTE COM A*
    def process_lidar(lidar_data):
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        
        # Filtro: Frente do carro
        frontend_points = points[(points[:, 0] > 0) & (points[:, 0] < 15) & (points[:, 1] > -1.5) & (points[:, 1] < 1.5)]
        
        if frontend_points.shape[0] > 5: # Se houver pontos suficientes (obstáculo real)
            min_dist = np.min(frontend_points[:, 0])
            
            if min_dist < 6.0:
                print(f"!!! EMERGÊNCIA: Obstáculo a {min_dist:.1f}m. Parando.")
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            
            elif min_dist < 15.0:
                print(f"Obstáculo detectado a {min_dist:.1f}m. Recalculando rota com A*...")
                
                # A* na prática: Buscar waypoint atual e o alvo à frente
                curr_wp = carla_map.get_waypoint(vehicle.get_location())
                next_wps = curr_wp.next(20.0)
                
                if next_wps:
                    target_wp = next_wps[0]
                    try:
                        # Executa a busca no grafo
                        path = nx.astar_path(graph, curr_wp.id, target_wp.id, 
                                             heuristic=lambda u, v: dist_heuristica(u, v, node_locations))
                        print(f"Caminho A* encontrado! Seguindo {len(path)} nós.")
                        tm.force_lane_change(vehicle, True) # Manobra de desvio
                    except nx.NetworkXNoPath:
                        print("A*: Nenhum caminho seguro encontrado.")

    lidar_sensor.listen(lambda data: process_lidar(data))

    # 7. LOOP DE EXECUÇÃO
    try:
        print("Agente Inteligente (A*) iniciado. Verifique a janela do CARLA.")
        while True:
            # Atualizar câmera espectadora para seguir o carro
            v_trans = vehicle.get_transform()
            spectator.set_transform(carla.Transform(v_trans.location + carla.Location(z=30), 
                                                 carla.Rotation(pitch=-90)))
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nLimpando atores...")
        lidar_sensor.destroy()
        vehicle.destroy()

if __name__ == '__main__':
    main()
