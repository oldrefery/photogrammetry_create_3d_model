# create_3d_model.py
DECREASE_IMAGE_QUALITY = True
TARGET_SIZE_DECREASE = 800

import os
import subprocess
# from pathlib import Path
import shutil
import psutil
import time
from datetime import datetime, timedelta
import sqlite3
import signal
import sys


def signal_handler(sig, frame):
    """Handle Ctrl+C"""
    print("\nGracefully shutting down... (Press Ctrl+C 3 times to force quit)")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def format_time(seconds):
    """Convert seconds to human readable format"""
    return str(timedelta(seconds=int(seconds)))


def check_database(database_path):
    """Check if database is valid"""
    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()[0]
        conn.close()
        return result == "ok"
    except:
        return False


def clean_database(output_dir):
    """Remove corrupted database"""
    database_path = os.path.join(output_dir, "database.db")
    if os.path.exists(database_path) and not check_database(database_path):
        print("Found corrupted database, removing...")
        os.remove(database_path)
        # Also delete the progress file, as the data may not be consistent
        progress_file = os.path.join(output_dir, "progress.txt")
        if os.path.exists(progress_file):
            os.remove(progress_file)
        return True
    return False


def run_cmd(cmd, desc=None, check_db=True):
    """Runs a command and prints the status with timing"""
    if desc:
        print(f"\n{desc}...")
    print(f"Running: {' '.join(cmd)}")

    start_time = time.time()
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Для обработки вывода в реальном времени
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Очищаем текущую строку и выводим новую информацию
                print(f"\r{output.strip()}", end='')
        
        # Получаем код возврата
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

    except subprocess.CalledProcessError as e:
        print(f"\nCommand failed with error code {e.returncode}")
        if e.output:
            print(f"Output: {e.output}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        raise

    elapsed = time.time() - start_time
    print(f"\nCompleted in: {format_time(elapsed)}")
    return elapsed


def process_feature_extractor_output(output):
    """Обработка вывода feature_extractor"""
    from tqdm import tqdm
    
    lines = output.split('\n')
    total_files = None
    pbar = None
    resolution_shown = False
    
    for line in lines:
        if "Processed file" in line:
            if not total_files:
                total_files = int(line.split('/')[1].split(']')[0])
                pbar = tqdm(total=total_files, desc="Extracting features")
            pbar.update(1)
            
            # Показываем разрешение только для первого файла
            if not resolution_shown and "Dimensions:" in line:
                dimensions = line.strip().split("Dimensions:")[1].strip()
                tqdm.write(f"Image dimensions: {dimensions}")
                resolution_shown = True
                
        elif "Elapsed time" in line:
            if pbar:
                pbar.close()
            print(line)


def filter_points_by_distance(dense_mvs, max_distance=0.5):
    """Filter points that are too far from the center"""
    try:
        import numpy as np
        from pathlib import Path
        import struct

        # Читаем MVS файл (бинарный формат)
        with open(dense_mvs, 'rb') as f:
            # Пропускаем заголовок
            f.seek(0, 2)  # перемещаемся в конец файла
            file_size = f.tell()
            f.seek(0)  # возвращаемся в начало

            # Находим блок с точками
            while f.tell() < file_size:
                print(f"\rProcessing points... {f.tell()/file_size*100:.1f}%", end='')

                chunk_type = struct.unpack('I', f.read(4))[0]
                chunk_size = struct.unpack('Q', f.read(8))[0]

                if chunk_type == 1:  # тип 1 - это точки
                    points = []
                    for _ in range(chunk_size):
                        x, y, z = struct.unpack('fff', f.read(12))
                        points.append([x, y, z])

                    points = np.array(points)
                    center = points.mean(axis=0)
                    distances = np.linalg.norm(points - center, axis=1)
                    mask = distances <= max_distance
                    filtered_points = points[mask]

                    # Создаем новый файл
                    filtered_mvs = str(Path(dense_mvs).with_suffix('.filtered.mvs'))
                    with open(filtered_mvs, 'wb') as out:
                        # Копируем заголовок
                        out.write(struct.pack('I', chunk_type))
                        out.write(struct.pack('Q', len(filtered_points)))
                        # Записываем отфильтрованные точки
                        for point in filtered_points:
                            out.write(struct.pack('fff', *point))

                    # Заменяем оригинальный файл
                    Path(filtered_mvs).rename(dense_mvs)
                    print(f"\rFiltered {len(points) - len(filtered_points)} distant points")
                    break
                else:
                    f.seek(chunk_size, 1)  # пропускаем неизвестный блок

    except Exception as e:
        print(f"\nWarning: Could not filter points: {e}")


def save_progress(output_dir, step, with_timestamp=True):
    """Save progress to a file with backup"""
    progress_file = os.path.join(output_dir, "progress.txt")
    backup_file = os.path.join(output_dir, "progress.txt.bak")

    try:
        # First save to backup file
        if os.path.exists(progress_file):
            shutil.copy2(progress_file, backup_file)

        # Then update main progress file
        with open(progress_file, "a") as f:
            if with_timestamp:
                f.write(f"{step}\t{datetime.now().isoformat()}\n")
            else:
                f.write(f"{step}\n")
    except Exception as e:
        print(f"Warning: Could not save progress for step {step}: {str(e)}")
        # Try to restore from backup
        if os.path.exists(backup_file):
            shutil.copy2(backup_file, progress_file)


def get_completed_steps(output_dir):
    """Get list of completed steps"""
    progress_file = os.path.join(output_dir, "progress.txt")
    backup_file = os.path.join(output_dir, "progress.txt.bak")

    completed = set()

    def read_progress(file_path):
        try:
            with open(file_path) as f:
                return {line.strip().split('\t')[0] for line in f}
        except Exception as e:
            print(f"Warning: Could not read progress from {file_path}: {str(e)}")
            return set()

    # Try main file first
    if os.path.exists(progress_file):
        completed = read_progress(progress_file)

    # If main file is empty or corrupted, try backup
    if not completed and os.path.exists(backup_file):
        completed = read_progress(backup_file)

    return completed


def clean_workspace(output_dir):
    """Cleaning the working directory"""
    print("\nCleaning workspace...")
    # Delete progress files
    for progress_file in ['progress.txt', 'progress.txt.bak']:
        path = os.path.join(output_dir, progress_file)
        if os.path.exists(path):
            os.remove(path)

    # Clean database directory
    database_dir = os.path.join(os.path.dirname(output_dir), "database")
    if os.path.exists(database_dir):
        for item in os.listdir(database_dir):
            path = os.path.join(database_dir, item)
            if os.path.isfile(path):
                os.remove(path)

    for item in ['sparse', 'dense']:
        path = os.path.join(output_dir, item)
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)


def convert_bin_to_txt(sparse_dir):
    """Convert COLMAP binary format to text format"""
    print("\nConverting COLMAP binary files to text...")
    sparse_path = os.path.join(sparse_dir, "0")
    if os.path.exists(sparse_path):
        try:
            # First, convert the model
            run_cmd([
                "colmap", "model_converter",
                "--input_path", sparse_path,
                "--output_path", sparse_path,
                "--output_type", "TXT"
            ], "Converting model to text format", check_db=False)

            # Then create undistorted images
            output_dir = os.path.dirname(sparse_dir)
            project_dir = os.path.dirname(output_dir)

            if DECREASE_IMAGE_QUALITY:
                image_dir = os.path.join(project_dir, "output", "processed_images")
            else:
                image_dir = os.path.join(project_dir, "images")

            run_cmd([
                "colmap", "image_undistorter",
                "--image_path", image_dir,
                "--input_path", sparse_path,
                "--output_path", os.path.join(output_dir, "dense"),
                "--output_type", "COLMAP"
            ], "Undistorting images", check_db=False)

        except Exception as e:
            print(f"Error during conversion: {str(e)}")
            raise


def check_platform():
    """Check if running on Apple Silicon"""
    import platform
    return platform.processor() == 'arm' and platform.system() == 'Darwin'


def create_mesh(project_dir):
    """Creates a textured mesh from COLMAP results using OpenMVS."""
    print("\n=== Starting mesh creation with OpenMVS ===")
    mesh_start_time = time.time()
    total_steps = 5  # Общее количество шагов
    current_step = 0

    output_dir = os.path.join(project_dir, "output")
    colmap_dir = os.path.join(output_dir, "sparse", "0")
    sparse_subdir = os.path.join(colmap_dir, "sparse")
    scene_mvs = os.path.join(output_dir, "scene.mvs")
    dense_mvs = os.path.join(output_dir, "scene_dense.mvs")
    mesh_mvs = os.path.join(output_dir, "scene_mesh.mvs")
    refined_mvs = os.path.join(output_dir, "scene_mesh_refined.mvs")
    textured_model = os.path.join(output_dir, "scene_mesh_textured")
    completed_steps = get_completed_steps(output_dir)

    try:
        for step in ["convert_to_mvs", "dense_reconstruction", "mesh_reconstruction", 
            "mesh_refinement", "texturing"]:
            current_step += 1
            print(f"\r[{current_step}/{total_steps}] Processing {step}...", end='')

        # 1. Scene Setup - Convert COLMAP to OpenMVS
        if "convert_to_mvs" not in completed_steps:
            print("\nStep 1: Setting up scene - Converting COLMAP to OpenMVS")
            # Create sparse subdirectory and copy files
            os.makedirs(sparse_subdir, exist_ok=True)
            for file in os.listdir(colmap_dir):
                if file.endswith((".txt", ".bin")):
                    shutil.copy2(os.path.join(colmap_dir, file), sparse_subdir)

            run_cmd([
                "/usr/local/bin/OpenMVS/InterfaceCOLMAP",
                "--working-folder", output_dir,
                "--input-file", colmap_dir,
                "--output-file", scene_mvs
            ], "Converting COLMAP to OpenMVS format")
            save_progress(output_dir, "convert_to_mvs")

        # 2. Dense Reconstruction
        if "dense_reconstruction" not in completed_steps and os.path.exists(scene_mvs):
            print("\nStep 2: Creating dense point cloud")
            run_cmd([
                "/usr/local/bin/OpenMVS/DensifyPointCloud",
                "--input-file", scene_mvs,
                "--output-file", dense_mvs,
                "--resolution-level", "1",
                "--number-views", "3",
                "--min-resolution", "640",
                "--max-resolution", "3200"
            ], "Creating dense point cloud")
            # Filter points after cloud creation
            if os.path.exists(dense_mvs):
                filter_points_by_distance(dense_mvs, max_distance=0.5)  # 50 см
            save_progress(output_dir, "dense_reconstruction")

        # 3. Mesh Reconstruction
        if "mesh_reconstruction" not in completed_steps and os.path.exists(dense_mvs):
            print("\nStep 3: Reconstructing mesh")
            # First copy the dense MVS file
            shutil.copy2(dense_mvs, mesh_mvs)
            run_cmd([
                "/usr/local/bin/OpenMVS/ReconstructMesh",
                "--input-file", dense_mvs,
                "--output-file", mesh_mvs,
                "--decimate", "0.5",
                "--thickness-factor", "0.8",
                "--quality-factor", "1.0",
                "--remove-spurious", "20"
            ], "Reconstructing mesh")
            save_progress(output_dir, "mesh_reconstruction")

        # 4. Mesh Refinement
        if "mesh_refinement" not in completed_steps and os.path.exists(mesh_mvs):
            print("\nStep 4: Refining mesh")
        try:
            refine_mesh_in_chunks(mesh_mvs, refined_mvs)
            save_progress(output_dir, "mesh_refinement")
        except Exception as e:
            print(f"\nError during mesh refinement: {str(e)}")
            # If it was not successful to process in parts, copy the original mesh
            shutil.copy2(mesh_mvs, refined_mvs)
                    # run_cmd([
            #     "/usr/local/bin/OpenMVS/RefineMesh",
            #     "--input-file", mesh_mvs,
            #     "--output-file", refined_mvs,
            #     "--resolution-level", "1",
            #     "--scales", "1",    
            #     "--scale-step", "0.8",
            #     "--max-face-area", "32" 
            # ], "Refining mesh")
            # save_progress(output_dir, "mesh_refinement")

        # 5. Texturing
        if "texturing" not in completed_steps:
            print("\nStep 5: Applying texture")
            # Copy dense MVS if refined doesn't exist
            if not os.path.exists(refined_mvs):
                shutil.copy2(dense_mvs, refined_mvs)

            run_cmd([
                "/usr/local/bin/OpenMVS/TextureMesh",
                "--input-file", refined_mvs,
                "--output-file", textured_model,
                "--export-type", "obj", #"glb",
                "--texture-size", "2048", #"4096",   # was "8192",
                "--outlier-threshold", "0.1"
            ], "Applying texture")
            save_progress(output_dir, "texturing")

        mesh_total_time = time.time() - mesh_start_time
        print(f"\nMesh creation completed in {format_time(mesh_total_time)}")

    except subprocess.CalledProcessError as e:
        print(f"Error executing OpenMVS command: {str(e)}")
        raise
    except Exception as e:
        print(f"\nError during mesh creation: {str(e)}")
        raise


def create_3d_model(image_dir, output_dir, clean=False):
    """Creates a 3D model from images"""
    total_start_time = time.time()
    print(f"Starting 3D reconstruction at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processing images from: {image_dir}")

    image_files = get_image_files(image_dir)
    if not image_files:
        raise ValueError(
            f"No valid images found in {image_dir}. "
            "Please ensure directory contains image files"
        )

    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory not found: {image_dir}")

    try:
        # Preprocess images
        if DECREASE_IMAGE_QUALITY:
            preprocess_images(image_dir, target_size=TARGET_SIZE_DECREASE)

        if clean or clean_database(output_dir):
            clean_workspace(output_dir)

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "sparse"), exist_ok=True)
        os.makedirs(os.path.join(os.path.dirname(output_dir), "database"), exist_ok=True)

        database_path = os.path.join(output_dir, "..", "database", "database.db")
        sparse_dir = os.path.join(output_dir, "sparse")
        completed_steps = get_completed_steps(output_dir)
        total_time = 0

        if DECREASE_IMAGE_QUALITY:
            image_dir =  os.path.join(os.path.dirname(image_dir), "output", "processed_images")
        print(f"Image directory: {image_dir}")


        # 1. Feature extraction
        if "feature_extraction" not in completed_steps or not os.path.exists(database_path):
            print("1. Feature extraction")
            total_time += run_cmd([
                "colmap", "feature_extractor",
                "--database_path", database_path,
                "--image_path", image_dir,
                "--ImageReader.camera_model", "PINHOLE",
                "--ImageReader.single_camera_per_folder", "0",
                "--SiftExtraction.use_gpu", "0",
                "--SiftExtraction.num_threads", "4",
                "--SiftExtraction.max_num_features", "16384",
                "--SiftExtraction.first_octave", "-1",
                "--SiftExtraction.peak_threshold", "0.002", #0.004
                "--SiftExtraction.edge_threshold", "15" #,
                # "--log_level", "2"
            ], "Extracting features", check_db=False)
            save_progress(output_dir, "feature_extraction")
        else:
            print("1. feature_extraction is already completed")


        # 2. Feature matching
        if "feature_matching" not in completed_steps and os.path.exists(database_path):
            print("2. Feature matching")
            total_time += run_cmd([
                "colmap", "exhaustive_matcher",
                "--database_path", database_path,
                "--SiftMatching.use_gpu", "0",
                "--SiftMatching.num_threads", str(psutil.cpu_count(logical=False))
            ], "Matching features")
            save_progress(output_dir, "feature_matching")
        else:
            print("2. feature_matching is already completed")

        # 3. Sparse reconstruction
        if "sparse_reconstruction" not in completed_steps and os.path.exists(database_path):
            print("3. Sparse reconstruction")
            os.makedirs(sparse_dir, exist_ok=True)
            total_time += run_cmd([
                "colmap", "mapper",
                "--database_path", database_path,
                "--image_path", image_dir,
                "--output_path", sparse_dir,
                "--Mapper.init_min_tri_angle", "8",
                "--Mapper.multiple_models", "0",
                "--Mapper.extract_colors", "1",
                "--Mapper.min_num_matches", "15",
                "--Mapper.ba_local_max_num_iterations", "50",
                "--Mapper.abs_pose_min_num_inliers", "15",
                "--Mapper.abs_pose_min_inlier_ratio", "0.25",
                "--Mapper.ba_global_images_ratio", "1.32",
                "--Mapper.ba_global_points_ratio", "1.32",
                "--Mapper.max_reg_trials", "3"
            ], "Running mapper")
            save_progress(output_dir, "sparse_reconstruction")

            # Ensure files are in the correct format
            if not os.path.exists(os.path.join(sparse_dir, "0", "cameras.txt")):
                run_cmd([
                    "colmap", "model_converter",
                    "--input_path", os.path.join(sparse_dir, "0"),
                    "--output_path", os.path.join(sparse_dir, "0"),
                    "--output_type", "TXT"
                ], "Converting model to text format")
        else:
            print("3. sparse_reconstruction is already completed")


        # 4. Dense reconstruction and further steps using OpenMVS
        print("4. Dense reconstruction using OpenMVS")
        create_mesh(os.path.dirname(output_dir))  # OpenMVS handles all further steps
        save_progress(output_dir, "creating mesh with openMVS is completed")

    except Exception as e:
        print(f"An error occurred during 3D model creation: {str(e)}")
        raise

    finally:
        total_time = time.time() - total_start_time
        print(f"\nTotal processing time: {format_time(total_time)}")


def get_image_files(directory):
    """Get list of image files with valid extensions"""
    import glob
    valid_extensions = ['.jpg', '.jpeg', '.png']
    return [f for f in glob.glob(os.path.join(directory, '*')) if os.path.splitext(f)[1].lower() in valid_extensions]


def preprocess_images(image_dir, target_size=1600):
   """Resize and standardize images."""
   import cv2
   import os
   from tqdm import tqdm
   
   processed_dir = os.path.join(os.path.dirname(image_dir), "output", "processed_images")
   os.makedirs(processed_dir, exist_ok=True)
   
   image_files = get_image_files(image_dir)
   processed_paths = []

   if not image_files:
       return None

   # Get the dimensions of the first image to output the information
   sample_img = cv2.imread(image_files[0])
   if sample_img is not None:
       height, width = sample_img.shape[:2]
       scale = min(target_size / width, target_size / height)
       new_width = int(width * scale)
       new_height = int(height * scale)
       print(f"Images will be resized from {width}x{height} to {new_width}x{new_height}")

   for img_path in tqdm(image_files, desc="Processing images", unit="img"):
       try:
           img = cv2.imread(img_path)
           if img is None:
               continue
               
           height, width = img.shape[:2]
           scale = min(target_size / width, target_size / height)
           new_width = int(width * scale)
           new_height = int(height * scale)
           
           img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
           filename = os.path.basename(img_path)
           output_path = os.path.join(processed_dir, filename)
           
           if cv2.imwrite(output_path, img):
               processed_paths.append(output_path)
               
       except Exception as e:
           tqdm.write(f"Error processing {img_path}: {str(e)}")
           
   return processed_dir if processed_paths else None


def refine_mesh_in_chunks(mesh_mvs, refined_mvs, chunk_size=1000000):
    """Split the mesh into parts and process them individually"""
    import open3d as o3d
    import numpy as np
    import tempfile
    from pathlib import Path
    
    print("\nSplitting mesh into chunks for processing...")
    
    # Загружаем меш
    mesh = o3d.io.read_triangle_mesh(mesh_mvs)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Define axis margins
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    
    # Split into subspaces
    x_chunks = 2  
    y_chunks = 2  
    x_ranges = np.linspace(x_min, x_max, x_chunks + 1)
    y_ranges = np.linspace(y_min, y_max, y_chunks + 1)
    
    temp_dir = tempfile.mkdtemp(prefix="mesh_chunks_")
    refined_parts = []
    
    try:
        # Process every chunk
        chunk_count = 0
        for i in range(len(x_ranges) - 1):
            for j in range(len(y_ranges) - 1):
                chunk_count += 1
                print(f"\nProcessing chunk {chunk_count} of {x_chunks * y_chunks}")
                
                # Select vertices in the current range
                mask = (vertices[:, 0] >= x_ranges[i]) & (vertices[:, 0] < x_ranges[i + 1]) & \
                       (vertices[:, 1] >= y_ranges[j]) & (vertices[:, 1] < y_ranges[j + 1])
                
                # Adding a little overlap
                overlap = 0.1  # 10% overlap
                overlap_x = (x_max - x_min) * overlap
                overlap_y = (y_max - y_min) * overlap
                
                extended_mask = (vertices[:, 0] >= x_ranges[i] - overlap_x) & \
                              (vertices[:, 0] < x_ranges[i + 1] + overlap_x) & \
                              (vertices[:, 1] >= y_ranges[j] - overlap_y) & \
                              (vertices[:, 1] < y_ranges[j + 1] + overlap_y)
                
                chunk_vertices = vertices[extended_mask]
                
                # Skip empty chunks
                if len(chunk_vertices) == 0:
                    continue
                
                # Create temporary files for the chunk
                chunk_file = os.path.join(temp_dir, f"chunk_{i}_{j}.mvs")
                refined_chunk = os.path.join(temp_dir, f"refined_chunk_{i}_{j}.mvs")
                
                # Save chunk 
                chunk_mesh = o3d.geometry.TriangleMesh()
                chunk_mesh.vertices = o3d.utility.Vector3dVector(chunk_vertices)
                o3d.io.write_triangle_mesh(chunk_file, chunk_mesh)
                
                try:
                    # Process chunk
                    run_cmd([
                        "/usr/local/bin/OpenMVS/RefineMesh",
                        "--input-file", chunk_file,
                        "--output-file", refined_chunk,
                        "--resolution-level", "1",
                        "--scales", "1",
                        "--scale-step", "0.8",
                        "--max-face-area", "32"
                    ], f"Refining chunk {chunk_count}")
                    
                    # Load processed chunk
                    refined_part = o3d.io.read_triangle_mesh(refined_chunk)
                    refined_parts.append(refined_part)
                    
                except Exception as e:
                    print(f"Error processing chunk {chunk_count}: {str(e)}")
                    continue
        
        # Merge all chunks
        print("\nMerging refined chunks...")
        final_mesh = o3d.geometry.TriangleMesh()
        for part in refined_parts:
            final_mesh += part
        
        # Save result
        print("Saving final refined mesh...")
        o3d.io.write_triangle_mesh(refined_mvs, final_mesh)
        
    finally:
        # Clean temp files
        import shutil
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print(f"!!!Starting the Project at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_dir = os.path.join(project_dir, "images")
    output_dir = os.path.join(project_dir, "output")

    save_progress(output_dir, "Start")

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Create 3D model from images')
    parser.add_argument('--clean', action='store_true', help='Clean workspace before starting')
    args = parser.parse_args()

    try:
        # Use clean flag from command line arguments
        create_3d_model(image_dir, output_dir, clean=args.clean)
        save_progress(output_dir, "Finish")
    except Exception as e:
        print(f"An error occurred: {str(e)}")