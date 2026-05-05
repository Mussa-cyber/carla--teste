import carla
import time
import random
import numpy as np
import networkx as nx

def dist_heuristica(n1, n2, locations):
    """Função h(n) para o A*: Distância euclidiana entre dois nós."""
    p1 = locations[n1]
    p2 = locations[n2]
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def main():
    # 1. CONEXÃO E SETUP DO MUNDO
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    carla_map = world.get_map()
    blueprint_library = world.get_blueprint_library()
    
    # 2. MAPEAMENTO DA CIDADE (GRAFO PARA A*)
    print("Mapeando estradas para o algoritmo A*...")
    graph = nx.DiGraph()
    node_locs = {}
    for seg in carla_map.get_topology():
        w1, w2 = seg[0], seg[1]
        node_locs[w1.id], node_locs[w2.id] = w1.transform.location, w2.transform.location
        graph.add_edge(w1.id, w2.id, weight=w1.transform.location.distance(w2.transform.location))

    # 3. SPAWN ALEATÓRIO COM PROTEÇÃO (90 CARROS NO TRÁFEGO)
    bp = blueprint_library.filter("model3")[0]
    spawn_points = carla_map.get_spawn_points()
    random.shuffle(spawn_points)
    
    vehicle = None
    for point in spawn_points:
        vehicle = world.try_spawn_actor(bp, point)
        if vehicle is not None:
            print(f"Sucesso! Veículo gerado em: {point.location}")
            break
    
    if vehicle is None:
        print("Erro: Não foi possível encontrar um ponto livre no tráfego denso.")
        return

    # 4. CONFIGURAÇÃO DO TRAFFIC MANAGER (ESTABILIDADE PARA APRESENTAÇÃO)
    tm = client.get_trafficmanager(8000)
    vehicle.set_autopilot(True, tm.get_port())
    
    # Velocidade reduzida para ~30km/h (Ajuste para apresentação)
    tm.global_percentage_speed_difference(50.0) 
    tm.set_global_distance_to_leading_vehicle(3.0)
    # Evita que o TM mude de faixa sozinho, deixando para a nossa lógica A*
    tm.auto_lane_change(vehicle, False) 

    # 5. SENSOR LIDAR (DETECÇÃO FRONTAL E LATERAL EXPANDIDA)
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '30')
    lidar_bp.set_attribute('rotation_frequency', '20')
    lidar_sensor = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=1.6, z=1.7)), attach_to=vehicle)

    def lidar_callback(data):
        # Converter dados do Lidar
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4')).reshape(-1, 4)
        
        # Filtro Frontal (Frenagem Crítica)
        front_collision = points[(points[:,0] > 0) & (points[:,0] < 7) & (np.abs(points[:,1]) < 1.5)]
        
        # Filtro Lateral (Detecção de faixas adjacentes para o A*)
        # Detecta em um retângulo de 15m à frente e 4.5m para cada lado
        lateral_obs = points[(points[:,0] > 0) & (points[:,0] < 15) & (np.abs(points[:,1]) < 4.5)]

        if len(front_collision) > 10:
            print("!!! EMERGÊNCIA: Objeto à frente. Travando rodas !!!")
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        
        elif len(lateral_obs) > 15:
            # Lógica A* simplificada: Se houver obstáculo, tenta trocar de faixa
            print("A*: Obstáculo lateral/frontal detectado. Planejando desvio...")
            curr_wp = carla_map.get_waypoint(vehicle.get_location())
            next_wps = curr_wp.next(20.0)
            
            if next_wps:
                try:
                    # Verifica se existe um caminho válido no grafo A*
                    nx.astar_path(graph, curr_wp.id, next_wps[0].id, 
                                  heuristic=lambda u, v: dist_heuristica(u, v, node_locs))
                    tm.force_lane_change(vehicle, True) # Força mudança para esquerda
                except nx.NetworkXNoPath:
                    print("A*: Caminho bloqueado em todas as direções.")

    lidar_sensor.listen(lidar_callback)

    # 6. LOOP PRINCIPAL: CÂMERA DE PERSEGUIÇÃO (AUTO-FOCUS)
    spectator = world.get_spectator()
    
    try:
        print("--- SISTEMA ATIVO ---")
        print("Acompanhe a janela do CARLA. O carro está sendo seguido automaticamente.")
        while True:
            # Pegar posição do carro
            v_trans = vehicle.get_transform()
            fwd = v_trans.get_forward_vector()
            
            # Calcular posição da câmera (12m atrás e 5m acima)
            cam_loc = v_trans.location - fwd * 12 + carla.Location(z=5)
            cam_rot = v_trans.rotation
            cam_rot.pitch = -20 # Inclinação para baixo
            
            # Atualizar o Spectator da interface
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))
            
            time.sleep(0.01) # Frequência de atualização da câmera

    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
    finally:
        print("Limpando ambiente para a próxima execução...")
        if 'lidar_sensor' in locals(): lidar_sensor.destroy()
        if 'vehicle' in locals(): vehicle.destroy()
        print("Recursos liberados.")

if __name__ == '__main__':
    main()
