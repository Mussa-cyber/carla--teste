import carla
import time
import random
import numpy as np
import networkx as nx

def dist_heuristica(n1, n2, locations):
    p1 = locations[n1]
    p2 = locations[n2]
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(15.0)
    world = client.get_world()
    carla_map = world.get_map()
    
    # SETUP DO GRAFO A*
    print("Mapeando estradas...")
    graph = nx.DiGraph()
    node_locs = {}
    for seg in carla_map.get_topology():
        w1, w2 = seg[0], seg[1]
        node_locs[w1.id], node_locs[w2.id] = w1.transform.location, w2.transform.location
        graph.add_edge(w1.id, w2.id, weight=w1.transform.location.distance(w2.transform.location))

    # SPAWN
    bp = world.get_blueprint_library().filter("model3")[0]
    # 1. Pegar todos os pontos de spawn e embaralhar a lista
    spawn_points = carla_map.get_spawn_points()
    random.shuffle(spawn_points)
    
    # 2. Tentar spawnar em cada ponto até encontrar um vazio
    vehicle = None
    for point in spawn_points:
        vehicle = world.try_spawn_actor(vehicle_bp, point)
        if vehicle is not None:
            print(f"Sucesso! Carro spawnado no ponto: {point.location}")
            break
    
    # 3. Verificação de segurança
    if vehicle is None:
        print("Erro: Não foi possível encontrar um ponto de spawn livre. Tente diminuir o tráfego.")
        return
    vehicle = world.spawn_actor(bp, spawn_pt)
    
    # SPECTATOR (A tela que já está aberta)
    spectator = world.get_spectator()

    # AUTOPILOTO E SENSOR
    tm = client.get_trafficmanager(8000)
    vehicle.set_autopilot(True, tm.get_port())
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
    lidar_sensor = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=1.6, z=1.7)), attach_to=vehicle)

    def lidar_callback(data):
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4')).reshape(-1, 4)
        # Filtro de colisão frontal
        front = points[(points[:,0] > 0) & (points[:,0] < 12) & (np.abs(points[:,1]) < 1.5)]
        if len(front) > 10:
            print("A*: Obstáculo detectado! Recalculando...")
            tm.force_lane_change(vehicle, True)

    lidar_sensor.listen(lidar_callback)

    try:
        print("Iniciando seguimento de câmera. Volte para a janela do CARLA.")
        while True:
            # Lógica de Câmera de Perseguição (Chase Cam)
            v_trans = vehicle.get_transform()
            fwd = v_trans.get_forward_vector()
            
            # Posiciona a câmera 12 metros atrás e 5 metros acima do carro
            cam_loc = v_trans.location - fwd * 12 + carla.Location(z=5)
            cam_rot = v_trans.rotation
            cam_rot.pitch = -20 # Inclina para baixo para ver o carro
            
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))
            time.sleep(0.01) # Alta frequência para suavidade

    except KeyboardInterrupt:
        pass
    finally:
        print("Limpando simulação...")
        lidar_sensor.destroy()
        vehicle.destroy()

if __name__ == '__main__':
    main()
