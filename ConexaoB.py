import carla
import time
import random
import numpy as np
import cv2 # Necessário para a janela extra

def main():
    # 1. SETUP DO MUNDO
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    carla_map = world.get_map()
    
    # 2. SPAWN DO VEÍCULO (Com proteção contra tráfego)
    bp = world.get_blueprint_library().filter("model3")[0]
    spawn_points = carla_map.get_spawn_points()
    random.shuffle(spawn_points)
    vehicle = None
    for point in spawn_points:
        vehicle = world.try_spawn_actor(bp, point)
        if vehicle is not None: break
    if vehicle is None: return

    # 3. CONFIGURAÇÃO DO TRAFFIC MANAGER
    tm = client.get_trafficmanager(8000)
    vehicle.set_autopilot(True, tm.get_port())
    tm.set_global_distance_to_leading_vehicle(1.0)
    tm.global_percentage_speed_difference(50.0) 
    tm.auto_lane_change(vehicle, False)

    # 4. SENSOR DE SEGMENTAÇÃO SEMÂNTICA
    # Este sensor colore objetos por categoria (Veículos = ID 10)
    sem_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
    sem_bp.set_attribute('image_size_x', '640')
    sem_bp.set_attribute('image_size_y', '480')
    sem_bp.set_attribute('fov', '110')
    sem_cam = world.spawn_actor(sem_bp, carla.Transform(carla.Location(x=1.6, z=1.7)), attach_to=vehicle)

    def camera_callback(image):
        # Converter imagem para array NumPy
        image.convert(carla.ColorConverter.CityCityPalette)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3] # Remover canal Alpha

        # LÓGICA DE DETECÇÃO (Visão Computacional Simples)
        # Vamos olhar para um retângulo no centro da imagem (onde estariam os carros à frente)
        # Em segmentação CityCityPalette, veículos aparecem em Azul Escuro/Preto (B=142, G=0, R=0)
        # Mas o ID puro no canal R é 10.
        
        # Recorte da zona central (ROI - Region of Interest)
        roi = array[300:450, 220:420]
        
        # Contar pixels que pertencem à categoria "Veículo" (Azul na paleta CityCity)
        # Na paleta CityCity, Veículos são (0, 0, 142) em RGB
        lower_blue = np.array([140, 0, 0])
        upper_blue = np.array([145, 0, 0])
        mask = cv2.inRange(roi, lower_blue, upper_blue)
        vehicle_pixels = cv2.countNonZero(mask)

        if vehicle_pixels > 500: # Se houver muitos pixels de carro à frente
            print(f"Visão: Obstáculo detectado ({vehicle_pixels}px). Tentando desviar...")
            tm.force_lane_change(vehicle, True)
            # Se estiver muito perto, trava
            if vehicle_pixels > 2500:
                vehicle.apply_control(carla.VehicleControl(brake=0.6))

        # Mostrar a janela do OpenCV para a apresentação
        cv2.imshow('Visao da IA - Segmentacao Semantica', array)
        cv2.waitKey(1)

    sem_cam.listen(camera_callback)

    # 5. LOOP DE CÂMERA SPECTATOR
    spectator = world.get_spectator()
    try:
        while True:
            v_trans = vehicle.get_transform()
            fwd = v_trans.get_forward_vector()
            cam_loc = v_trans.location - fwd * 12 + carla.Location(z=5)
            spectator.set_transform(carla.Transform(cam_loc, v_trans.rotation))
            time.sleep(0.02)
    except KeyboardInterrupt: pass
    finally:
        cv2.destroyAllWindows()
        sem_cam.destroy()
        vehicle.destroy()

if __name__ == '__main__':
    main()
