import carla
import random
import time
import cv2
import os
import numpy as np

# Caminho configurado conforme solicitado
OUTPUT_DIR = r"C:\Users\moisesmanganhel\Documents\Carla-project"

# Cria a pasta caso ela não exista
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    actor_list = []
    
    try:
        # 1. Conecta ao servidor (Certifique-se de que o CarlaUE4.exe está rodando)
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        print("Conectado ao CARLA com sucesso!")

        # 2. Spawn do Veículo (Tesla Model 3)
        bp_vehicle = blueprint_library.find('vehicle.tesla.model3')
        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp_vehicle, spawn_point)
        actor_list.append(vehicle)
        print(f"Veículo criado em: {spawn_point.location}")

        # 3. Spawn de alguns pedestres para o mundo ficar vivo
        print("Spawnando pedestres...")
        for i in range(5):
            bp_walker = random.choice(blueprint_library.filter('walker.pedestrian.*'))
            spawn_point_walker = random.choice(world.get_map().get_spawn_points())
            walker = world.spawn_actor(bp_walker, spawn_point_walker)
            actor_list.append(walker)

        # 4. Sensor de Câmera (RGB)
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '800')
        cam_bp.set_attribute('image_size_y', '600')
        cam_bp.set_attribute('fov', '90')
        
        # Posiciona a câmera no topo do veículo
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
        actor_list.append(camera)

        # Função de callback para salvar a imagem no seu caminho especificado
        def save_image(image):
            # Converte os dados brutos para array numpy
            i = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            i = i.reshape((image.height, image.width, 4))
            i = i[:, :, :3] # Remove canal alpha (transparência)
            
            # Salva o arquivo no caminho correto
            file_path = os.path.join(OUTPUT_DIR, f"frame_{image.frame}.jpg")
            cv2.imwrite(file_path, i)
            print(f"Frame {image.frame} salvo em: {file_path}")

        # Inicia a escuta da câmera
        camera.listen(lambda image: save_image(image))

        print("Simulação rodando. Verifique a pasta 'Carla-project' para ver as fotos.")
        print("Pressione Ctrl+C no terminal para parar.")
        
        # Mantém o script em loop infinito
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nInterrupção detectada pelo usuário.")

    finally:
        print("Destruindo atores...")
        for actor in actor_list:
            actor.destroy()
        print("Limpeza concluída. Até logo!")

if __name__ == '__main__':
    main()