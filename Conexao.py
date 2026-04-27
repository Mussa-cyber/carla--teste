import carla
import time
import numpy as np

def main():
    # 1. CONEXÃO E SETUP INICIAL
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # 2. INSTANCIAR VEÍCULO
    vehicle_bp = blueprint_library.filter("model3")[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # 3. CONFIGURAR TRAFFIC MANAGER (Navegação base)
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(False)
    vehicle.set_autopilot(True, tm.get_port())

    # 4. CONFIGURAR SENSOR LIDAR
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '30')
    lidar_bp.set_attribute('rotation_frequency', '10')
    
    lidar_transform = carla.Transform(carla.Location(x=1.6, z=1.7))
    lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

    # 5. LÓGICA DO AGENTE (CÉREBRO)
    def on_obstacle_detected(lidar_data):
        # Converter dados brutos do Lidar para numpy array
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        
        # Filtro: Apenas pontos na frente (x>0) e na largura da faixa (-1 < y < 1)
        frontend_points = points[(points[:, 0] > 0) & (points[:, 1] > -1) & (points[:, 1] < 1)]
        
        if frontend_points.shape[0] > 0:
            min_dist = np.min(frontend_points[:, 0])
            
            # --- HIERARQUIA DE DECISÃO ---
            if min_dist < 5.0:
                # Prioridade 1: Segurança (Travar)
                print(f"!!! PERIGO: {min_dist:.1f}m. FREANDO !!!")
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                
            elif min_dist < 15.0:
                # Prioridade 2: Conforto/Navegação (Mudar de faixa)
                print(f"Obstáculo a {min_dist:.1f}m. Mudando de faixa...")
                tm.force_lane_change(vehicle, True) # True = esquerda
            
            else:
                pass # Distância segura

    # 6. INICIAR MONITORAMENTO
    lidar_sensor.listen(lambda data: on_obstacle_detected(data))

    # 7. EXECUÇÃO
    try:
        print("Agente autônomo iniciado. Pressione Ctrl+C para parar.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nEncerrando simulação...")
    finally:
        # LIMPEZA OBRIGATÓRIA
        lidar_sensor.destroy()
        vehicle.destroy()
        print("Recursos liberados. Conexão encerrada.")

if __name__ == '__main__':
    main()