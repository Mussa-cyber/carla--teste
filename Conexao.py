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
    # 1. CONEXÃO E SETUP
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    carla_map = world.get_map()
    blueprint_library = world.get_blueprint_library()
    
    # 2. MAPEAMENTO DA ESTRADA (GRAFO A*)
    print("Mapeando estradas para o algoritmo A*...")
    graph = nx.DiGraph()
    node_locs = {}
    for seg in carla_map.get_topology():
        w1, w2 = seg[0], seg[1]
        node_locs[w1.id], node_locs[w2.id] = w1.transform.location, w2.transform.location
        graph.add_edge(w1.id, w2.id, weight=w1.transform.location.distance(w2.transform.location))

    # 3. SPAWN ALEATÓRIO COM PROTEÇÃO CONTRA COLISÃO
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
        print("Erro: Nenhum ponto de spawn livre encontrado.")
        return

    # 4. CONFIGURAÇÃO DO TRAFFIC MANAGER (TM)
    tm = client.get_trafficmanager(8000)
    vehicle.set_autopilot(True, tm.get_port())
    
    # Configurações para apresentação
    tm.global_percentage_speed_difference(50.0) # Reduz velocidade média
    tm.set_global_distance_to_leading_vehicle(2.5) # Mantém distância segura
    tm.auto_lane_change(vehicle, False) # Desativa IA padrão para usarmos o Lidar + A*
    
    # 5. SENSOR LIDAR (DETECÇÃO FRONTAL E LATERAL)
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '30')
    lidar_bp.set_attribute('rotation_frequency', '25')
    lidar_sensor = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=1.6, z=1.7)), attach_to=vehicle)

    def lidar_callback(data):
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4')).reshape(-1, 4)
        
        # Filtro Frontal Estrito (Obstáculo na minha faixa)
        front_obs = points[(points[:,0] > 0) & (points[:,0] < 12) & (np.abs(points[:,1]) < 1.0)]
        
        # Filtro Faixa Esquerda (Verifica se há carros ao lado antes de mudar)
        # Analisa a área de 1.5m a 4.5m à esquerda do carro
        left_lane_busy = points[(points[:,0] > -2) & (points[:,0] < 10) & (points[:,1] > 1.5) & (points[:,1] < 4.5)]

        if len(front_obs) > 15:
            print("Obstáculo detectado à frente.")
            
            # Decisão de Mudança de Faixa
            if len(left_lane_busy) < 5:
                print("A*: Caminho livre à esquerda. Iniciando ultrapassagem...")
                tm.force_lane_change(vehicle, True) # Força a troca para a esquerda
            else:
                print("A*: Faixa esquerda ocupada. Reduzindo velocidade para evitar colisão.")
                # Aplica freio se estiver muito perto e bloqueado
                if np.min(front_obs[:, 0]) < 5.0:
                    vehicle.apply_control(carla.VehicleControl(brake=0.6))

    lidar_sensor.listen(lidar_callback)

    # 6. LOOP PRINCIPAL: CÂMERA DE PERSEGUIÇÃO
    spectator = world.get_spectator()
    
    try:
        print("--- SISTEMA DE NAVEGAÇÃO ATIVO ---")
        while True:
            # Rastreamento do Spectator
            v_trans = vehicle.get_transform()
            fwd = v_trans.get_forward_vector()
            
            # Câmera posicionada para ver a manobra e o Lidar trabalhando
            cam_loc = v_trans.location - fwd * 12 + carla.Location(z=5)
            cam_rot = v_trans.rotation
            cam_rot.pitch = -20
            
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nFinalizado.")
    finally:
        print("Limpando atores...")
        if 'lidar_sensor' in locals(): lidar_sensor.destroy()
        if 'vehicle' in locals(): vehicle.destroy()
        print("Recursos liberados.")

if __name__ == '__main__':
    main()
