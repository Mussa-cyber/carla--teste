import carla
import time
import numpy as np
import networkx as nx

def dist_heuristica(n1, n2, locations):
    p1 = locations[n1]
    p2 = locations[n2]
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def main():
    # 1. CONEXÃO
    client = carla.Client("localhost", 2000)
    client.set_timeout(15.0)
    world = client.get_world()
    carla_map = world.get_map()
    
    # 2. MAPEAMENTO PARA A* (Otimizado)
    print("Gerando grafo de navegação...")
    graph = nx.DiGraph()
    node_locations = {}
    for segment in carla_map.get_topology():
        wp1, wp2 = segment[0], segment[1]
        node_locations[wp1.id], node_locations[wp2.id] = wp1.transform.location, wp2.transform.location
        graph.add_edge(wp1.id, wp2.id, weight=wp1.transform.location.distance(wp2.transform.location))

    # 3. SPAWN E FOCO IMEDIATO
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("model3")[0]
    spawn_point = carla_map.get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    
    # Pegar o Spectator (a câmera da tela já aberta)
    spectator = world.get_spectator()
    print("Veículo spawnado. Iniciando rastreamento de câmera...")

    # 4. AUTOPILOTO E SENSORES
    tm = client.get_trafficmanager(8000)
    vehicle.set_autopilot(True, tm.get_port())
    
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '30')
    lidar_sensor = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=1.6, z=1.7)), attach_to=vehicle)

    # 5. CALLBACK DO LIDAR (Lógica A*)
    def process_lidar(lidar_data):
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        frontend = points[(points[:, 0] > 0) & (points[:, 0] < 12) & (np.abs(points[:, 1]) < 1.5)]
        
        if len(frontend) > 10:
            curr_wp = carla_map.get_waypoint(vehicle.get_location())
            next_wps = curr_wp.next(15.0)
            if next_wps:
                try:
                    nx.astar_path(graph, curr_wp.id, next_wps[0].id, 
                                  heuristic=lambda u, v: dist_heuristica(u, v, node_locations))
                    print("A*: Obstáculo à frente! Ajustando trajetória.")
                    tm.force_lane_change(vehicle, True)
                except:
                    vehicle.apply_control(carla.VehicleControl(brake=1.0))

    lidar_sensor.listen(lambda data: process_lidar(data))

    # 6. LOOP DE SEGUIMENTO (O "PULO DO GATO")
    try:
        while True:
            # Pegar a transformação atual do carro
            v_transform = vehicle.get_transform()
            
            # Calcular a posição da câmera (Atrás e no Alto)
            # -10 metros no eixo X (atrás) e +5 metros no eixo Z (cima)
            # Usamos o vetor de direção do carro para a câmera estar sempre atrás dele
            forward_vec = v_transform.get_forward_vector()
            camera_location = v_transform.location - forward_vec * 10 + carla.Location(z=5)
            
            # Ajustar a rotação para olhar para o carro
            camera_rotation = v_transform.rotation
            camera_rotation.pitch = -20 # Olhar um pouco para baixo
            
            # Aplicar ao Spectator da tela aberta
            spectator.set_transform(carla.Transform(camera_location, camera_rotation))
            
            time.sleep(0.02) # ~50 FPS de atualização de câmera
            
    except KeyboardInterrupt:
        print("\nFinalizando...")
    finally:
        lidar_sensor.destroy()
        vehicle.destroy()
