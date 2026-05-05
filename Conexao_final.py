import carla
import time
import random
import numpy as np

def main():
    # 1. CONEXÃO E SETUP
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    carla_map = world.get_map()
    blueprint_library = world.get_blueprint_library()

    # 2. SPAWN ALEATÓRIO COM PROTEÇÃO CONTRA COLISÃO
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

    # 3. CONFIGURAÇÃO DO TRAFFIC MANAGER (TM)
    tm = client.get_trafficmanager(8000)
    vehicle.set_autopilot(True, tm.get_port())

    tm.global_percentage_speed_difference(50.0)
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.auto_lane_change(vehicle, False)  # Desativa IA padrão — controlamos nós

    # 4. SENSOR LIDAR
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '30')
    lidar_bp.set_attribute('rotation_frequency', '25')
    lidar_sensor = world.spawn_actor(
        lidar_bp,
        carla.Transform(carla.Location(x=1.6, z=1.7)),
        attach_to=vehicle
    )

    def lidar_callback(data):
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4')).reshape(-1, 4)

        # Filtro frontal: objetos à frente entre 1.5m e 12m, na largura da faixa
        front_obs = points[
            (points[:, 0] > 1.5) & (points[:, 0] < 12) &
            (np.abs(points[:, 1]) < 1.0) &
            (points[:, 2] > -1.2)
        ]

        # Filtro faixa ESQUERDA: Y positivo no CARLA = esquerda do veículo
        left_lane_busy = points[
            (points[:, 0] > -2) & (points[:, 0] < 10) &
            (points[:, 1] > 1.8) & (points[:, 1] < 4.5) &
            (points[:, 2] > -1.0)
        ]

        # Filtro faixa DIREITA: Y negativo no CARLA = direita do veículo
        right_lane_busy = points[
            (points[:, 0] > -2) & (points[:, 0] < 10) &
            (points[:, 1] > -4.5) & (points[:, 1] < -1.8) &
            (points[:, 2] > -1.0)
        ]

        if len(front_obs) > 20:
            # Prioridade 1: tentar ultrapassar pela ESQUERDA
            if len(left_lane_busy) < 3:
                print("Caminho livre à ESQUERDA! Executando manobra...")
                tm.force_lane_change(vehicle, True)   # True = esquerda

            # Prioridade 2: tentar ultrapassar pela DIREITA
            elif len(right_lane_busy) < 3:
                print("Esquerda ocupada. Caminho livre à DIREITA! Executando manobra...")
                tm.force_lane_change(vehicle, False)  # False = direita

            # Ambas ocupadas: travar
            else:
                print(
                    f"Ambas as faixas ocupadas "
                    f"(esq: {len(left_lane_busy)} pts | dir: {len(right_lane_busy)} pts). "
                    f"A travar..."
                )
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.5))

    lidar_sensor.listen(lidar_callback)

    # 5. LOOP PRINCIPAL: CÂMERA DE PERSEGUIÇÃO
    spectator = world.get_spectator()

    try:
        print("--- SISTEMA DE NAVEGAÇÃO ATIVO ---")
        while True:
            v_trans = vehicle.get_transform()
            fwd = v_trans.get_forward_vector()

            cam_loc = v_trans.location - fwd * 12 + carla.Location(z=5)
            cam_rot = v_trans.rotation
            cam_rot.pitch = -20

            spectator.set_transform(carla.Transform(cam_loc, cam_rot))
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nFinalizado.")
    finally:
        print("Limpando atores...")
        if 'lidar_sensor' in locals():
            lidar_sensor.destroy()
        if 'vehicle' in locals():
            vehicle.destroy()
        print("Recursos liberados.")

if __name__ == '__main__':
    main()
